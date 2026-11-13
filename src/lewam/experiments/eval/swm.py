from collections import deque
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ...datasets.swm import _COLUMNS
from ...utils.normalize import Normalizer
from ...utils.spec import action_entry, spec_fields
from .base import EvalResult


class _ChunkPolicy:
    """Executes a model's predicted action chunk open-loop, replanning once the buffer
    for that env drains. `model.chunk_len` comes from the algorithm itself, so eval never
    needs its own copy of the chunk length. If the model carries `norm_stats` (attached by
    the training loop / restored from its checkpoint, see utils/normalize.py), observations
    are normalized and predicted actions are unnormalized the same way training saw them.

    Which raw env fields to read is driven entirely by `model.conditioning` (the same
    `from` keys SWMSource windows during training, see datasets/swm.py's `_COLUMNS`), so
    this works unchanged across envs/algorithms -- e.g. PushT's `observation.env_state`
    (env info key `state`) vs. TwoRoom's `observation` (env info key `observation`).

    """

    def __init__(self, model):
        self.model = model
        self.env = None
        self._buffers = None
        self._obs_hist = None
        self.obs_len = getattr(model, "obs_len", 1)
        # a field's offsets may be sparse, so each samples its own out of the dense
        # obs_len history, matching the union row layout the selectors were built against
        self._offsets = dict(getattr(model, "field_offsets", None) or {})
        self.action_stream = model.action_stream
        self.action_field = model.action_field
        if self.action_stream is None:
            raise ValueError(
                "executing an action chunk needs a predicted stream that carries actions; "
                "mark one with `role: action`, or evaluate this model with CEM planning."
            )
        self.execute_len = getattr(model, "execute_len", model.chunk_len)
        stats = getattr(model, "norm_stats", None)
        device = next(model.parameters()).device
        method = getattr(model, "norm_method", "mean_std")
        self.normalizer = Normalizer(stats, method=method).to(device) if stats else None
        self._fields = sorted({
            f for name, spec in model.conditioning.items() for f in spec_fields(name, spec)
        })
        # fields whose window includes the "last" frame: at eval that slot is the env goal
        self._last_fields = getattr(model, "_last_fields", set())

    def set_env(self, env):
        self.env = env
        self._buffers = [deque() for _ in range(env.num_envs)]
        self._obs_hist = [deque(maxlen=self.obs_len) for _ in range(env.num_envs)]

    def get_action(self, info_dict, **kwargs):
        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None:
            for i, flush in enumerate(needs_flush):
                if flush:
                    self._buffers[i].clear()
                    self._obs_hist[i].clear()

        for i in range(len(self._obs_hist)):
            frame = {
                field: info_dict[_COLUMNS.get(field, field)][i][-1] for field in self._fields
            }
            self._obs_hist[i].append(frame)

        replan = [i for i, buf in enumerate(self._buffers) if len(buf) == 0]
        if replan:
            device = next(self.model.parameters()).device

            def stacked(i, field):
                hist = list(self._obs_hist[i])
                hist = [hist[0]] * (self.obs_len - len(hist)) + hist
                offs = self._offsets.get(field)
                if offs is not None:
                    hist = [hist[self.obs_len - 1 + o] for o in offs]
                return np.stack([frame[field] for frame in hist])

            obs = {}
            for field in self._fields:
                raw = torch.as_tensor(np.stack([stacked(i, field) for i in replan])).to(device)
                if field in self._last_fields:
                    # the "last" slot is the env goal, appended after the history so the model's
                    # end-relative selector finds it. swm keys it 'goal' for pixels, 'goal_<col>' otherwise.
                    col = _COLUMNS.get(field, field)
                    key = "goal" if col == "pixels" else f"goal_{col}"
                    if key not in info_dict:
                        raise KeyError(
                            f"'{field}' conditions on the episode-final ('last') frame, sourced "
                            f"at rollout from info['{key}'], but the env exposes "
                            f"{sorted(info_dict)}. Use a goal-emitting env variant "
                            f"(e.g. swm/TwoRoomGoalImage-v1, see lewam/envs)."
                        )
                    goal = np.stack([np.asarray(info_dict[key][i][-1]) for i in replan])
                    raw = torch.cat([raw, torch.as_tensor(goal).to(device).unsqueeze(1)], dim=1)
                if raw.ndim == 5 and raw.shape[-1] in (1, 3):   # (B,T,H,W,C) image field
                    raw = raw.permute(0, 1, 4, 2, 3)
                else:
                    raw = raw.float()
                    if self.normalizer is not None:
                        raw = self.normalizer.normalize(field, raw)
                obs[field] = raw
            chunk = self.model.predict(obs)
            if isinstance(chunk, dict):
                chunk = chunk[self.action_stream]
            if self.normalizer is not None:
                chunk = self.normalizer.unnormalize(self.action_field, chunk)
            chunk = chunk.cpu().numpy()
            for row, env_i in enumerate(replan):
                self._buffers[env_i].extend(chunk[row][: self.execute_len])

        action = np.stack([self._buffers[i].popleft() for i in range(len(self._buffers))])
        return action.astype(np.float32)


class _WorldModelCost:
    """Adapts a predict-state LeWAM to swm's Costable interface for CEM planning.

    swm's CEMSolver calls `get_cost(info_dict, action_candidates)` where each info value
    is expanded to (bs, num_samples, ...) and candidates are (bs, num_samples, horizon,
    action_dim). This flattens the sample grid, builds each conditioning field's history
    window (repeating the current frame, as the model was trained to tolerate at episode
    start), delegates to `model.get_cost`, and reshapes the cost back to (bs, num_samples).
    Observations are normalized and goals mapped the same way training saw them.

    The comparison itself is selected by the eval config's `cost` block:

        cost:
          type: state          # default: L2(unnormalized final state, info goal_state)

        cost:
          type: encoded_goal   # JEPA-style: L2(final latent, encoder(info[goal_key]))
          encoder: video       # name in model.encoders; encodes the goal observation
          goal_key: goal       # info key holding the goal image/obs (default "goal")
          stream: future_video # optional: predicted stream to roll; defaults to the
                               # spec's only predicted stream

    """

    def __init__(self, model, cost_cfg=None):
        self.model = model
        self.obs_len = getattr(model, "obs_len", 1)
        stats = getattr(model, "norm_stats", None)
        device = next(model.parameters()).device
        self.device = device
        method = getattr(model, "norm_method", "mean_std")
        self.normalizer = Normalizer(stats, method=method).to(device) if stats else None
        # a world model predicts state, so its action is a CONDITIONING stream and
        # model.action_field is None
        cond_action = action_entry(model.conditioning)
        if cond_action is None:
            raise ValueError(
                "CEM planning drives the world model with candidate actions, but no "
                "conditioning stream carries them. Mark one with `role: action`."
            )
        self.action_field = cond_action[1]
        self._fields = sorted({
            spec.get("from", name)
            for name, spec in model.conditioning.items()
            if spec.get("from", name) != self.action_field
        })

        cost_cfg = cost_cfg or {}
        self.cost_type = cost_cfg.get("type", "state")
        assert self.cost_type in ("state", "encoded_goal"), f"unknown cost type '{self.cost_type}'"
        self.goal_key = cost_cfg.get("goal_key", "goal")
        self.stream = cost_cfg.get("stream")
        if self.stream is None:
            assert len(model.predict_spec) == 1, (
                f"model predicts {sorted(model.predict_spec)}; set cost.stream to pick one"
            )
            self.stream = next(iter(model.predict_spec))
        self.goal_encoder = None
        if self.cost_type == "encoded_goal":
            name = cost_cfg.get("encoder")
            encoders = getattr(model, "encoders", {})
            assert name in encoders, (
                f"cost.encoder '{name}' not among the model's encoders {sorted(encoders)}"
            )
            run_encoder = getattr(model, "run_encoder", None)
            self.goal_encoder = (
                (lambda obs: run_encoder(name, obs)) if run_encoder else encoders[name]
            )

    def parameters(self):
        return self.model.parameters()

    def _window(self, frame):
        # (N, dim) current frame -> (N, obs_len, dim) by repetition
        return frame.unsqueeze(1).expand(-1, self.obs_len, -1).contiguous()

    def _image_window(self, raw):
        # (N, [T,] H, W, C) -> (N, obs_len, C, H, W); the env carries a time axis
        n, hwc = raw.shape[0], raw.shape[-3:]
        frames = raw.reshape(n, -1, *hwc).permute(0, 1, 4, 2, 3)   # (N, T, C, H, W)
        if frames.shape[1] < self.obs_len:
            pad = frames[:, :1].expand(-1, self.obs_len - frames.shape[1], -1, -1, -1)
            frames = torch.cat([pad, frames], dim=1)
        return frames[:, -self.obs_len:].contiguous()

    def get_cost(self, info_dict, action_candidates):
        bs, num_samples = action_candidates.shape[:2]
        n = bs * num_samples
        cand = action_candidates.reshape(n, *action_candidates.shape[2:]).to(self.device).float()
        # CEM samples in raw env units; the model conditions on the normalized action
        # stream it trained on. Executed actions stay raw.
        if self.normalizer is not None:
            cand = self.normalizer.normalize(self.action_field, cand)

        obs = {}
        for field in self._fields:
            # env info carries raw column names (PushT: `state`), same mapping as _ChunkPolicy
            raw = torch.as_tensor(info_dict[_COLUMNS.get(field, field)]).to(self.device)
            raw = raw.reshape(n, *raw.shape[2:])            # collapse the (bs, num_samples) grid
            if raw.shape[-1] in (1, 3) and raw.ndim >= 4:   # image; last 3 dims are (H, W, C)
                obs[field] = self._image_window(raw)
            else:
                frame = raw.reshape(n, -1).float()
                if self.normalizer is not None:
                    frame = self.normalizer.normalize(field, frame)
                obs[field] = self._window(frame)

        past = info_dict.get("past_actions")
        if past is not None:
            past = torch.as_tensor(past).to(self.device).float()
            past = past.reshape(n, *past.shape[2:])
            if self.normalizer is not None:
                past = self.normalizer.normalize(self.action_field, past)

        if self.model.predict_spec[self.stream].get("encoder"):
            final = self.model.rollout_latent(obs, cand, stream=self.stream, past_actions=past)
        else:
            final = self.model.rollout_final_state(obs, cand, stream=self.stream)

        if self.cost_type == "encoded_goal":
            if self.goal_key not in info_dict:
                raise KeyError(
                    f"encoded_goal cost needs a goal image at info['{self.goal_key}'], but the "
                    f"env exposes {sorted(info_dict)}. PushT emits 'goal'; TwoRoom does not in "
                    f"episodic eval, so it needs dataset goals or a goal-rendering env."
                )
            goal = torch.as_tensor(info_dict[self.goal_key]).to(self.device)
            goal = goal.reshape(n, *goal.shape[2:])
            if goal.shape[-1] in (1, 3) and goal.ndim >= 4:   # goal image -> (N, 1, C, H, W)
                goal = self._image_window(goal)[:, -1:]
            goal_emb = self.goal_encoder(goal).reshape(n, -1)
            cost = torch.linalg.vector_norm(final.reshape(n, -1) - goal_emb, dim=-1)
            return cost.reshape(bs, num_samples)

        # goal_state is in raw env units; the model predicts normalized, so invert
        # with the predicted stream's own field key
        goal = torch.as_tensor(info_dict["goal_state"]).reshape(n, -1).to(self.device).float()
        state_field = self.model.predict_spec[self.stream].get("from", self.stream)
        if self.normalizer is not None:
            final = self.normalizer.unnormalize(state_field, final)
        cost = torch.linalg.vector_norm(final[:, : goal.shape[-1]] - goal, dim=-1)
        return cost.reshape(bs, num_samples)


def _single_env_bounds(space):
    """Per-env (act_dim,) bounds from a possibly BATCHED vector-env Box, whose rows all
    carry the same single-env bounds. The solver can be asked to replan a SUBSET of envs
    (others already terminated), so the batched space's leading dim need not match the
    solve batch -- never reshape by it.
    """
    low, high = np.asarray(space.low), np.asarray(space.high)
    if low.ndim > 1:
        low, high = low[0], high[0]
    return low.reshape(-1), high.reshape(-1)


def _box_cem_solver_cls():
    """CEMSolver subclass whose INITIAL distribution covers the env's Box action space.

    swm's CEMSolver starts every plan at mean=0, std=var_scale -- correct for the
    symmetric [-1, 1]-style delta actions swm envs default to, but on an absolute
    coordinate Box (PushT absolute pixel targets in [0, 512]) every first-iteration
    candidate then lands within ~var_scale pixels of the ORIGIN, i.e. the top-left
    corner, and the elite update can never escape it: the agent beelines to (0, 0)
    every episode.

    For a non-symmetric (absolute) Box we widen the initial std to a quarter of the box
    span and recenter the initial mean, with two choices of center:
      - default: the box center;
      - center_on_agent=True: the agent's CURRENT position, read raw (same pixel units as
        the action) from info_dict (see _agent_xy), tiled across every horizon step. This is
        the faithful translation of le-wm's mean=0 "stay put" prior to an absolute action
        space -- le-wm's zero delta is "no movement", i.e. target the current position.
    Only padded (all-zero) rows of a warm start are recentered; real warm rows are kept.
    Symmetric spaces are left exactly as upstream, so TwoRoom behavior is unchanged.
    Deferred import: stable_worldmodel is an optional eval-time dependency.
    """
    import numpy as np
    import torch
    from gymnasium.spaces import Box
    from stable_worldmodel.solver.cem import CEMSolver

    class _BoxCEM(CEMSolver):
        center_on_agent = False   # set by _BoxCEMSolver; see class docstring

        def _agent_xy(self, info_dict):
            """Current agent position (n_envs, 2) in raw action units, or None.

            `state` is preferred over `pos_agent`: in dataset-driven eval the env is
            teleported to a demo state AFTER reset, and only info keys that exist as
            dataset columns are refreshed -- `state` is, `pos_agent` is not, so the
            latter is stale (the discarded random reset) for the first solve.
            """
            for key in ("state", "pos_agent", "proprio"):
                v = info_dict.get(key)
                if v is None:
                    continue
                v = v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))
                return v.reshape(v.shape[0], -1, v.shape[-1])[:, -1, :2].to(self._dtype)
            return None

        def solve(self, info_dict, init_action=None):
            # aligned with the mean rows, for init_action_distrib which gets no info_dict
            self._agent_pos = self._agent_xy(info_dict) if self.center_on_agent else None
            out = super().solve(info_dict, init_action)
            # unclamped Gaussian elites on an absolute [0, 512] space drift off-board, to
            # targets the env PD controller chases off-screen and training never saw
            space = getattr(self, "_action_space", None)
            if (
                isinstance(space, Box)
                and np.isfinite(space.low).all()
                and np.isfinite(space.high).all()
            ):
                acts = out["actions"]
                low, high = _single_env_bounds(space)
                low = torch.as_tensor(low, dtype=acts.dtype)
                high = torch.as_tensor(high, dtype=acts.dtype)
                reps = acts.shape[-1] // low.shape[-1]      # action_block tiling
                out["actions"] = torch.clamp(acts, low.repeat(reps), high.repeat(reps))
            return out

        def init_action_distrib(self, n_envs, actions=None):
            mean, var = super().init_action_distrib(n_envs, actions)
            space = getattr(self, "_action_space", None)
            if not (
                isinstance(space, Box)
                and np.isfinite(space.low).all()
                and np.isfinite(space.high).all()
            ):
                return mean, var
            low, high = _single_env_bounds(space)
            low = torch.as_tensor(low, dtype=mean.dtype).unsqueeze(0).expand(mean.shape[0], -1)
            high = torch.as_tensor(high, dtype=mean.dtype).unsqueeze(0).expand(mean.shape[0], -1)
            reps = mean.shape[-1] // low.shape[-1]     # action_block tiling
            span = ((high - low) / 4).repeat(1, reps)
            agent = getattr(self, "_agent_pos", None)
            # trust the agent position only if it is inside the box in RAW units
            use_agent = (
                agent is not None and agent.shape == low.shape
                and bool((agent >= low - 1).all()) and bool((agent <= high + 1).all())
            )
            if use_agent:
                center = agent.repeat(1, reps).to(mean.dtype)
            else:
                center = ((low + high) / 2).repeat(1, reps)
                if not (center.abs() > span).any():
                    return mean, var
            blank = mean.abs().sum(-1, keepdim=True) == 0
            mean = torch.where(blank, center[:, None, :].expand_as(mean).to(mean.device), mean)
            var = var * span[:, None, :].to(var.device)
            return mean, var

    return _BoxCEM


def _BoxCEMSolver(*args, center_on_agent=False, **kwargs):
    solver = _box_cem_solver_cls()(*args, **kwargs)
    solver.center_on_agent = bool(center_on_agent)
    return solver


def _read_videos(video_dir, max_videos):
    import imageio

    paths = sorted(
        p for pat in ("episode_*.mp4", "env_*.mp4") for p in Path(video_dir).glob(pat)
    )[:max_videos]
    videos = {}
    for path in paths:
        frames = np.stack([f for f in imageio.get_reader(str(path))])
        videos[path.stem] = frames.transpose(0, 3, 1, 2)
    return videos


class SWMEval:
    """Runs the trained policy as open-loop action chunks in a stable-worldmodel world
    (e.g. PushT) and reports success rate + a handful of rollout videos.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, model):
        import stable_worldmodel as swm

        import lewam.envs  # noqa: F401  (registers LeWAM env variants with gym)

        cfg = self.cfg
        world = swm.World(
            cfg.env_name,
            num_envs=cfg.num_envs,
            image_shape=tuple(cfg.image_shape),
            max_episode_steps=cfg.max_episode_steps,
        )
        if model.action_stream is not None:
            world.set_policy(_ChunkPolicy(model))
        else:
            world.set_policy(self._planning_policy(model))

        with ExitStack() as stack:
            video_dir = cfg.get("video_dir") or stack.enter_context(TemporaryDirectory())
            if cfg.get("dataset_eval"):
                results = self._dataset_evaluate(world, video_dir)
            else:
                # EPISODIC eval: valid only when the demos cover the env's full init range
                results = world.evaluate(episodes=cfg.episodes, seed=cfg.seed, video=video_dir)
            videos = _read_videos(video_dir, cfg.get("max_videos", 4))
        world.close()

        return EvalResult(
            metrics={"success_rate": results["success_rate"]}, videos=videos
        )

    def _history_policy_cls(self):
        """WorldModelPolicy that repairs the eval-time conditioning gap: swm's World only
        exposes the CURRENT frame, so the cost would repeat it obs_len times ("nothing has
        moved"), and the rollout would zero-pad the pre-context action slots -- neither of
        which training ever saw. This subclass tracks each env's frames and executed raw
        actions and injects (a) an observation history at `history_spacing` env steps per
        model step (= action_block for frameskip-trained models) and (b) the last
        executed action blocks under info key 'past_actions'. Deferred import: swm is an
        optional eval-time dependency.
        """
        from stable_worldmodel.policy import WorldModelPolicy

        class _HistoryPolicy(WorldModelPolicy):
            def __init__(self, *args, history_spacing=1, obs_len=1, **kwargs):
                super().__init__(*args, **kwargs)
                self._spacing = int(history_spacing)
                self._obs_len = int(obs_len)
                self._hist = None       # per-env deque of (frame, raw_action_before_frame)
                self._acts = None       # per-env deque of executed raw actions

            def set_env(self, env):
                super().set_env(env)
                n = getattr(env, "num_envs", 1)
                depth = (self._obs_len - 1) * self._spacing + 1
                self._hist = [deque(maxlen=depth) for _ in range(n)]
                block = self.cfg.action_block
                self._acts = [deque(maxlen=(self._obs_len - 1) * block) for _ in range(n)]

            def get_action(self, info_dict, **kwargs):
                flush = info_dict.pop("_needs_flush", None)
                if flush is not None:
                    for i, fl in enumerate(flush):
                        if fl:
                            self._hist[i].clear()
                            self._acts[i].clear()

                px = np.asarray(info_dict["pixels"])
                for i in range(len(self._hist)):
                    frame = px[i][-1] if px[i].ndim > 3 else px[i]
                    self._hist[i].append(np.asarray(frame))

                def stacked(i):
                    h = list(self._hist[i])
                    idxs = [len(h) - 1 - k * self._spacing for k in range(self._obs_len - 1, -1, -1)]
                    return np.stack([h[max(j, 0)] for j in idxs])

                def past_blocks(i):
                    block = self.cfg.action_block
                    need = (self._obs_len - 1) * block
                    acts = list(self._acts[i])
                    pad = [np.zeros_like(acts[0]) if acts else np.zeros(2, dtype=np.float32)
                           for _ in range(need - len(acts))]
                    flat = np.stack(pad + acts) if (pad or acts) else np.zeros((need, 2), np.float32)
                    return flat.reshape(self._obs_len - 1, -1)

                info = dict(info_dict)
                if flush is not None:
                    info["_needs_flush"] = flush
                info["pixels"] = np.stack([stacked(i) for i in range(len(self._hist))])
                info["past_actions"] = np.stack([past_blocks(i) for i in range(len(self._hist))]).astype(np.float32)

                action = super().get_action(info, **kwargs)
                for i in range(len(self._acts)):
                    if not np.isnan(action[i]).any():
                        self._acts[i].append(np.asarray(action[i], dtype=np.float32))
                return action

        return _HistoryPolicy

    def _dataset_evaluate(self, world, video_dir):
        """le-wm's dataset-driven eval (le-wm eval.py): seed each vectorized env from a
        random expert-demo state (`_set_state`) and target the demo state `goal_offset`
        steps ahead in that same demo (`_set_goal_state`), with `eval_budget` steps to
        reach it. Short-horizon goal-reaching on in-distribution states -- the metric
        le-wm reports -- unlike the episodic fixed-goal full task above. swm requires one
        env per start, so `num_envs` is the number of evals (le-wm: num_eval=50).
        """
        from stable_worldmodel.data import HDF5Dataset
        from stable_worldmodel.data.utils import get_cache_dir

        from lewam.datasets.swm import SWMSource

        cfg = self.cfg.dataset_eval
        datasets_dir = get_cache_dir(sub_folder="datasets")
        if not (datasets_dir / f"{cfg.name}.h5").exists():
            SWMSource._download(cfg.hf_repo, cfg.name, datasets_dir)

        # the h5 stores 224x224 but the policy sees cfg.image_shape renders; area
        # interpolation matches what SWMSource applies at train time
        size = tuple(self.cfg.image_shape)

        def resize_pixels(steps):
            px = steps.get("pixels")
            if px is not None and tuple(px.shape[-2:]) != size:
                steps["pixels"] = (
                    torch.nn.functional.interpolate(px.float(), size=size, mode="area")
                    .round().clamp(0, 255).to(px.dtype)
                )
            return steps

        dataset = HDF5Dataset(cfg.name, transform=resize_pixels, cache_dir=cfg.get("cache_dir"))

        starts = [
            (ep, t)
            for ep, length in enumerate(np.asarray(dataset.lengths))
            for t in range(int(length) - cfg.goal_offset)
        ]
        g = np.random.default_rng(self.cfg.seed)
        picks = g.choice(len(starts), size=self.cfg.num_envs, replace=False)
        episodes_idx, start_steps = map(list, zip(*(starts[i] for i in picks)))

        return world.evaluate(
            dataset=dataset,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            goal_offset=cfg.goal_offset,
            eval_budget=cfg.eval_budget,
            callables=[
                {"method": "_set_state", "args": {"state": {"value": "state"}}},
                {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
            ],
            video=video_dir,
        )

    def _planning_policy(self, model):
        from stable_worldmodel.policy import PlanConfig, WorldModelPolicy

        plan = self.cfg.plan
        device = next(model.parameters()).device
        solver = _BoxCEMSolver(
            _WorldModelCost(model, self.cfg.get("cost")),
            num_samples=plan.get("num_samples", 300),
            n_steps=plan.get("n_steps", 30),
            topk=plan.get("topk", 30),
            var_scale=plan.get("var_scale", 1.0),
            device=str(device),
            seed=self.cfg.get("seed", 1234),
            center_on_agent=plan.get("center_on_agent", False),
            # plan all vectorized envs in one batched solve (default 1 serializes them)
            batch_size=plan.get("batch_size", self.cfg.get("num_envs", 1)),
        )
        config = PlanConfig(
            horizon=plan.horizon,
            receding_horizon=plan.receding_horizon,
            history_len=getattr(model, "obs_len", 1),
            action_block=plan.get("action_block", 1),
        )
        obs_len = getattr(model, "obs_len", 1)
        if obs_len > 1:
            return self._history_policy_cls()(
                solver=solver, config=config,
                history_spacing=plan.get("history_spacing", plan.get("action_block", 1)),
                obs_len=obs_len,
            )
        return WorldModelPolicy(solver=solver, config=config)
