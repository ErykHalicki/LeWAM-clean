"""The eval-side normalizers must mirror the dataset's method and stats exactly:
observations normalized on the way in, model outputs unnormalized on the way out,
using each stream's own field key."""

import numpy as np
import pytest
import torch

from lewam.experiments.eval.swm import _ChunkPolicy, _WorldModelCost


class _StubWM(torch.nn.Module):
    """Predict-state stub with PushT's field names; echoes what it was fed."""

    def __init__(self, stats):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.obs_len = 2
        self.conditioning = {
            "state_context": {"from": "observation.env_state", "index": "-1..0"},
            "action": {"from": "action", "index": "0"},
        }
        self.predict_spec = {"future_state": {"from": "observation.env_state"}}
        self.norm_stats = stats
        self.norm_method = "mean_std"
        self.seen = {}

    def rollout_final_state(self, obs, action_candidates, stream=None):
        self.seen["obs"] = obs
        self.seen["cand"] = action_candidates
        return obs["observation.env_state"][:, -1]


def test_wm_cost_normalizes_in_and_out_with_stream_field_keys():
    dim = 7
    stats = {
        "observation.env_state": {
            "mean": [100.0] * dim, "std": [50.0] * dim,
            "min": [0.0] * dim, "max": [512.0] * dim,
        },
        "action": {
            "mean": [0.5, -0.5], "std": [2.0, 4.0],
            "min": [-1.0, -1.0], "max": [1.0, 1.0],
        },
    }
    model = _StubWM(stats)
    cost_fn = _WorldModelCost(model)

    bs, num_samples, horizon = 1, 3, 4
    state = np.full((bs, num_samples, dim), 100.0, dtype=np.float32)
    goal = np.full((bs, num_samples, dim), 150.0, dtype=np.float32)
    info = {"state": state, "goal_state": goal}
    cand = torch.full((bs, num_samples, horizon, 2), 0.5)

    cost = cost_fn.get_cost(info, cand)

    seen = model.seen["cand"]
    assert torch.allclose(seen[..., 0], torch.zeros_like(seen[..., 0]))
    assert torch.allclose(seen[..., 1], torch.full_like(seen[..., 1], 0.25))
    assert torch.allclose(model.seen["obs"]["observation.env_state"], torch.zeros(3, 2, dim))
    # the predicted state was unnormalized back to raw units before the goal
    # comparison, 50 per dim over 7 dims
    expected = np.linalg.norm(np.full(dim, 50.0))
    assert cost.shape == (bs, num_samples)
    assert torch.allclose(cost, torch.full((bs, num_samples), float(expected)), atol=1e-4)


def test_wm_cost_encoded_goal_compares_in_latent_space():
    dim = 7
    stats = {
        "observation.env_state": {
            "mean": [100.0] * dim, "std": [50.0] * dim,
            "min": [0.0] * dim, "max": [512.0] * dim,
        },
        "action": {
            "mean": [0.0, 0.0], "std": [1.0, 1.0],
            "min": [-1.0, -1.0], "max": [1.0, 1.0],
        },
    }
    model = _StubWM(stats)
    model.encoders = {"latent": lambda goal: goal.reshape(goal.shape[0], -1)[:, :dim] * 2.0}
    cost_fn = _WorldModelCost(model, {"type": "encoded_goal", "encoder": "latent", "goal_key": "goal"})

    bs, num_samples, horizon = 1, 3, 4
    state = np.full((bs, num_samples, dim), 100.0, dtype=np.float32)
    goal_obs = np.full((bs, num_samples, dim), 3.0, dtype=np.float32)
    info = {"state": state, "goal": goal_obs}
    cand = torch.zeros((bs, num_samples, horizon, 2))

    cost = cost_fn.get_cost(info, cand)

    # final latent is 0 (stub echoes the normalized state); encoder(goal) = 6 per dim
    expected = np.linalg.norm(np.full(dim, 6.0))
    assert cost.shape == (bs, num_samples)
    assert torch.allclose(cost, torch.full((bs, num_samples), float(expected)), atol=1e-4)


def _tiny_wm_model():
    from omegaconf import OmegaConf

    from lewam.algorithms import build_algorithm

    rope = {"time": {"share": 0.5, "period": "auto"},
            "height": {"share": 0.125, "period": "auto"},
            "width": {"share": 0.125, "period": "auto"},
            "seq": {"share": 0.25, "period": "auto"}}
    vit_rope = {"time": {"share": 0.5, "period": "auto"},
                "height": {"share": 0.25, "period": "auto"},
                "width": {"share": 0.25, "period": "auto"}}
    cfg = OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {"rope": rope, "model_dim": 64, "depth": 2, "num_heads": 4},
        "num_flow_steps": 2,
        # img_size is the H of the frames the test below feeds; frames covers the -1..0 window
        "encoders": {"vit": {"type": "vit", "patch_size": 8, "dim": 64, "depth": 1, "num_heads": 4,
                             "rope": vit_rope, "img_size": 16, "frames": 2}},
        "conditioning": {
            "pixel_context": {"from": "observation.images.pixels", "encoder": "vit", "dim": 64,
                              "index": "-1..0", "fps": 10, "grid": [2, 2]},
            "action": {"from": "action", "dim": 2, "index": "0", "fps": 10},
        },
        "predict": {
            "future_video": {"from": "observation.images.pixels", "encoder": "vit", "type": "flow",
                             "dim": 64, "index": "1", "fps": 10, "grid": [2, 2]},
        },
        "losses": [{"type": "flow", "stream": "future_video", "weight": 1.0}],
    })
    model = build_algorithm(cfg)
    model.norm_stats = None
    model.norm_method = "mean_std"
    return model


def test_wm_cost_encoded_goal_latent_rollout_with_images():
    torch.manual_seed(0)
    model = _tiny_wm_model()
    cost_fn = _WorldModelCost(
        model, {"type": "encoded_goal", "encoder": "vit", "goal_key": "goal", "stream": "future_video"}
    )

    bs, num_samples, horizon, H = 2, 3, 4, 16
    # the swm env carries a time axis: pixels arrive (bs, num_samples, T, H, W, C)
    pixels = np.random.randint(0, 256, (bs, num_samples, 1, H, H, 3), dtype=np.uint8)
    goal = np.random.randint(0, 256, (bs, num_samples, H, H, 3), dtype=np.uint8)
    info = {"pixels": pixels, "goal": goal}
    cand = torch.randn(bs, num_samples, horizon, 2)

    cost = cost_fn.get_cost(info, cand)

    assert cost.shape == (bs, num_samples)
    assert torch.isfinite(cost).all()
    assert (cost >= 0).all()


class _StubGoalPolicy(torch.nn.Module):
    """Pixel policy stub whose conditioning includes the episode-final ("last") slot."""

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.obs_len = 2
        self.chunk_len = 3
        self.execute_len = 3
        self.action_stream = "action"
        self.action_field = "action"
        self.norm_stats = None
        self.conditioning = {
            "pixel_context": {"from": "observation.images.pixels", "index": "-1..0"},
            "goal": {"from": "observation.images.pixels", "index": "last", "via": "cross"},
        }
        self._last_fields = {"observation.images.pixels"}
        self.seen = {}

    def predict(self, obs):
        self.seen["obs"] = obs
        b = next(iter(obs.values())).shape[0]
        return torch.zeros(b, self.chunk_len, 2)


class _FakeEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs


def test_chunk_policy_appends_env_goal_for_last_fields():
    model = _StubGoalPolicy()
    policy = _ChunkPolicy(model)
    policy.set_env(_FakeEnv(2))

    H = 8
    pixels = np.random.randint(0, 256, (2, 1, H, H, 3), dtype=np.uint8)
    goal = np.random.randint(0, 256, (2, 1, H, H, 3), dtype=np.uint8)

    action = policy.get_action({"pixels": pixels, "goal": goal})
    assert action.shape == (2, 2)

    seen = model.seen["obs"]["observation.images.pixels"]
    assert seen.shape == (2, model.obs_len + 1, 3, H, H)
    assert torch.equal(seen[:, -1], torch.as_tensor(goal[:, -1]).permute(0, 3, 1, 2))
    assert torch.equal(seen[:, 0], torch.as_tensor(pixels[:, -1]).permute(0, 3, 1, 2))


def test_chunk_policy_missing_goal_key_raises():
    model = _StubGoalPolicy()
    policy = _ChunkPolicy(model)
    policy.set_env(_FakeEnv(1))
    pixels = np.random.randint(0, 256, (1, 1, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(KeyError, match="goal"):
        policy.get_action({"pixels": pixels})


def test_box_cem_solver_centers_absolute_action_spaces():
    # upstream CEM inits at mean=0/std=1, which on PushT's absolute [0, 512] pixel
    # action Box puts every candidate at the top-left corner
    pytest.importorskip("stable_worldmodel")
    import numpy as np
    from gymnasium.spaces import Box

    from lewam.experiments.eval.swm import _box_cem_solver_cls

    class _Cfg:
        horizon = 4
        action_block = 1

    def make(low, high):
        solver = _box_cem_solver_cls()(model=torch.nn.Linear(1, 1), num_samples=8, topk=2)
        solver.configure(
            action_space=Box(low=low, high=high, shape=(3, 2), dtype=np.float32),
            n_envs=3, config=_Cfg(),
        )
        return solver

    solver = make(0.0, 512.0)
    warm = torch.zeros(3, 4, 2)
    warm[:, 0] = 100.0
    mean, var = solver.init_action_distrib(3, warm)
    assert torch.allclose(mean[:, 0], torch.full((3, 2), 100.0))
    assert torch.allclose(mean[:, 1:], torch.full((3, 3, 2), 256.0))
    assert torch.allclose(var, torch.full((3, 4, 2), 128.0))

    sym = make(-1.0, 1.0)
    mean, var = sym.init_action_distrib(3, torch.zeros(3, 4, 2))
    assert torch.equal(mean, torch.zeros(3, 4, 2))
    assert torch.equal(var, torch.ones(3, 4, 2))
