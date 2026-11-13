"""V-JEPA 2-AC on the LeWAM batch contract: a frozen video encoder, an action-conditioned
block-causal predictor over its patch tokens, trained with Meta's teacher-forced +
autoregressive-rollout L1 on layer-normed latents, and planned with CEM against a goal
frame's latent.

Spec (one camera stream, one action stream, one predicted stream):

    conditioning:
      scene_context: {from: <camera>, encoder: vjepa, via: tokens, dim: D, grid: [h, w],
                      index: "-1..0", raw_index: "-12, 0", fps: ...}
      action:        {from: action, role: action, dim: A, index: "-12..35"}
    predict:
      scene_future:  {from: <camera>, encoder: vjepa, type: direct, dim: D, grid: [h, w],
                      index: "1..3", raw_index: "12, 24, 36", fps: ...}

The action rows span every transition between consecutive frames (context and future),
`sub` raw actions per transition; the predictor's action token for frame t is the
flattened `sub x A` block executed between frames t and t+1. Planning searches that same
space, `len(future)` tokens deep, so a plan is a `chunk_len = len(future) * sub` action
chunk.
"""

import torch
import torch.nn.functional as F

from ..planning.cem import cem
from ..utils.spec import action_entry, parse_index, spec_fields
from .ac_predictor import build_action_only_predictor
from .predictive_model import PredictiveModel


class VJEPA2AC(PredictiveModel):
    # a world model: no predicted stream carries actions, so rollouts plan instead of execute
    action_stream = None

    def __init__(self, cfg):
        super().__init__(cfg)
        assert self.action_stream is None, "vjepa2_ac predicts latents, not actions"

        cond = dict(cfg.conditioning)
        action = action_entry(cond)
        if action is None:
            raise ValueError("vjepa2_ac needs a conditioning stream with `role: action`")
        self.action_cond, self.action_field = action
        visual = [n for n, s in cond.items() if n != self.action_cond]
        if len(visual) != 1 or cond[visual[0]].get("encoder") is None:
            raise ValueError(
                f"vjepa2_ac conditions on exactly one encoded camera stream plus the action "
                f"stream; got {sorted(cond)}"
            )
        self.ctx_stream = visual[0]
        if len(cfg.predict) != 1:
            raise ValueError(f"vjepa2_ac predicts exactly one stream; got {sorted(cfg.predict)}")
        self.future_stream = next(iter(cfg.predict))
        ctx, fut = cond[self.ctx_stream], cfg.predict[self.future_stream]
        if fut.get("type") != "direct":
            raise ValueError(f"stream '{self.future_stream}' must be `type: direct`")
        if spec_fields(self.ctx_stream, ctx) != spec_fields(self.future_stream, fut) \
                or ctx.get("encoder") != fut.get("encoder"):
            raise ValueError("context and future streams must read one field through one encoder")
        self.field = spec_fields(self.ctx_stream, ctx)[0]
        self.encoder_name = ctx["encoder"]
        if self._raw_per_step[self.ctx_stream] != 1:
            raise ValueError("the camera encoder must yield one latent step per raw frame "
                             "(vjepa2: per_frame: true, frames_per_step: 1)")
        root = self.encoders[self._encoder_chains[self.encoder_name][0]]
        if not root.frozen:
            raise ValueError("vjepa2_ac trains the predictor only; set the encoder `frozen: true`")

        self.frames_ctx = len(parse_index(ctx["index"]))
        self.frames_future = len(parse_index(fut["index"]))
        self.num_frames = self.frames_ctx + self.frames_future
        transitions = self.num_frames - 1
        rows = len(parse_index(cond[self.action_cond]["index"]))
        if rows % transitions:
            raise ValueError(
                f"{rows} action rows do not split over the {transitions} frame transitions"
            )
        self.sub = rows // transitions
        self.action_dim = int(cond[self.action_cond]["dim"])
        self.action_embed_dim = self.sub * self.action_dim
        self.past_steps = self.frames_ctx - 1
        self.horizon = self.frames_future
        self.chunk_len = self.horizon * self.sub
        self.execute_len = int(cfg.get("execute_len", self.chunk_len))
        assert 1 <= self.execute_len <= self.chunk_len

        grid = tuple(int(g) for g in ctx.get("grid", (1, 1)))
        self.grid = grid
        self.tokens_per_frame = grid[0] * grid[1]
        self.latent_dim = int(ctx["dim"])

        loss = dict(cfg.get("loss") or {})
        self.auto_steps = int(loss.get("auto_steps", transitions))
        if not 1 <= self.auto_steps <= transitions:
            raise ValueError(f"auto_steps must be within [1, {transitions}]")
        self.normalize_reps = bool(loss.get("normalize_reps", True))
        self.predictor = self._build_predictor(cfg)

    def _build_predictor(self, cfg):
        if not hasattr(self, "grid"):
            return None        # the base constructor calls this before the spec is read
        p = dict(cfg.predictor)
        patch = 16
        return build_action_only_predictor(
            img_size=(self.grid[0] * patch, self.grid[1] * patch),
            patch_size=patch,
            num_frames=2 * self.num_frames,
            tubelet_size=2,
            embed_dim=self.latent_dim,
            predictor_embed_dim=int(p["dim"]),
            depth=int(p["depth"]),
            num_heads=int(p["heads"]),
            drop_path_rate=float(p.get("drop_path", 0.0)),
            use_rope=True,
            is_frame_causal=True,
            action_embed_dim=self.action_embed_dim,
        )

    # ---- encoding ---------------------------------------------------------------

    def _norm(self, z):
        return F.layer_norm(z, (z.size(-1),)) if self.normalize_reps else z

    def encode_frames(self, frames):
        """(B, T, C, H, W) uint8 -> (B, T, P, D) layer-normed latents."""
        z = self.run_encoder(self.encoder_name, frames)
        B, T = frames.shape[:2]
        z = z.view(B, T, self.tokens_per_frame, -1)
        return self._norm(z)

    def _action_tokens(self, rows):
        """(B, steps*sub, A) raw action rows -> (B, steps, sub*A) one token per transition."""
        B, n, A = rows.shape
        if n % self.sub or A != self.action_dim:
            raise ValueError(
                f"action rows {tuple(rows.shape)} do not form {self.sub}x{self.action_dim} tokens"
            )
        return rows.reshape(B, n // self.sub, self.sub * A)

    def _step(self, z_flat, actions):
        out = self.predictor(z_flat, actions)
        return self._norm(out)

    # ---- training ---------------------------------------------------------------

    def _encode_stream(self, name, spec, batch):
        """(B, T, P, D) layer-normed latents of one stream, from the encoding cache when
        the batch carries its keys, else from pixels."""
        z = self._encode(name, spec, batch)
        return self._norm(z.view(z.shape[0], -1, self.tokens_per_frame, z.shape[-1]))

    def loss(self, batch):
        P = self.tokens_per_frame
        z = torch.cat([
            self._encode_stream(self.ctx_stream, self.conditioning[self.ctx_stream], batch),
            self._encode_stream(self.future_stream, self.predict_spec[self.future_stream], batch),
        ], dim=1)                                                         # (B, F, P, D)
        a = self._action_tokens(
            self._raw(self.action_cond, self.conditioning[self.action_cond], batch).float()
        )                                                                 # (B, F-1, sub*A)
        Fn = z.shape[1]
        target = z[:, 1:].flatten(1, 2)

        # teacher forcing: ground-truth frames 0..F-2 predict frames 1..F-1
        z_tf = self._step(z[:, :-1].flatten(1, 2), a)
        loss_tf = (z_tf - target).abs().mean()

        # rollout: ground-truth frame 0 + the predicted frame 1, then each prediction fed
        # back, `auto_steps` frames deep
        cur = torch.cat([z[:, 0], z_tf[:, :P]], dim=1)
        for n in range(1, self.auto_steps):
            nxt = self._step(cur, a[:, : n + 1])[:, -P:]
            cur = torch.cat([cur, nxt], dim=1)
        z_ar = cur[:, P:]
        loss_ar = (z_ar - target[:, : self.auto_steps * P]).abs().mean()

        latent = {
            self.ctx_stream: z[:, : self.frames_ctx].flatten(1, 2),
            self.future_stream: z[:, self.frames_ctx:].flatten(1, 2),
        }
        return {
            "loss": loss_tf + loss_ar,
            "loss/tf": loss_tf.detach(),
            "loss/ar": loss_ar.detach(),
            **self._variance_logs(latent),
        }

    def optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_gradient_checkpointing(self, enabled):
        super().set_gradient_checkpointing(enabled)
        self.predictor.use_activation_checkpointing = bool(enabled)

    def summary(self):
        pred = sum(p.numel() for p in self.predictor.parameters())
        return {**super().summary(), "params/predictor": pred, "chunk_len": self.chunk_len}

    # ---- inference --------------------------------------------------------------

    @torch.no_grad()
    def predict(self, obs):
        """Teacher-forced future latents: obs carries the context frames and every action
        row (history and future) -> (N, len(future) * P, D)."""
        z = self.encode_frames(obs[self.field])
        a = self._action_tokens(obs[self.action_field].float())
        return self.rollout_latent(z, a[:, : self.past_steps], a[:, self.past_steps:],
                                   keep_all=True)

    @torch.no_grad()
    def rollout_latent(self, ctx, past_actions, candidates, keep_all=False):
        """ctx (N, frames_ctx, P, D) latents, past_actions (N, past_steps, sub*A) or None,
        candidates (N, H, sub*A) -> the latent `H` frames ahead (N, P, D), or every
        predicted frame (N, H*P, D) with `keep_all`.

        Each step re-runs the predictor over the whole sequence (context, then the
        predictions so far) and keeps its last frame, as Meta's rollout does.
        """
        N, H = candidates.shape[:2]
        P = self.tokens_per_frame
        if past_actions is None:
            past_actions = candidates.new_zeros(N, self.past_steps, candidates.shape[-1])
        if past_actions.shape[1] != self.past_steps:
            raise ValueError(
                f"{self.frames_ctx} context frames need {self.past_steps} past action "
                f"tokens, got {past_actions.shape[1]}"
            )
        cur = ctx.flatten(1, 2)
        preds = []
        for h in range(H):
            acts = torch.cat([past_actions, candidates[:, : h + 1]], dim=1)
            nxt = self._step(cur, acts)[:, -P:]
            preds.append(nxt)
            cur = torch.cat([cur, nxt], dim=1)
        return torch.cat(preds, dim=1) if keep_all else preds[-1]

    @torch.no_grad()
    def plan(self, frames, goal, past_rows=None, cfg=None, generator=None):
        """CEM over action chunks toward a goal latent.

        frames (N, frames_ctx, C, H, W) uint8; goal (N, P, D) latent in `encode_frames`
        space; past_rows (N, past_steps*sub, A) normalized actions already executed, or
        None (history at the normalized centre, as the dataset pads an episode start).
        Returns (N, chunk_len, A) normalized actions. Energy is Meta's: L1 between the
        final rolled-out latent and the goal. `cfg.chunk` bounds the candidates scored
        per predictor pass.
        """
        cfg = dict(cfg or {})
        piece = int(cfg.pop("chunk", 256))
        # the search space: `chunk` = every raw action of every step (H x sub*A, what the
        # predictor consumes); `hold` = one action per latent step held for its sub rows;
        # `linear` = one knot per latent step, linearly interpolated from the last executed
        # action. Only the low-dimensional spaces propose the smooth chunks the predictor was
        # trained on: measured on held-out PnP-T, chunk/sigma 1 planned 43 cm of
        # tool error, linear/sigma 0.1 5.2 cm (persistence 7.0 cm)
        space = str(cfg.pop("space", "linear"))
        z = self.encode_frames(frames)
        N = z.shape[0]
        past = None if past_rows is None else self._action_tokens(past_rows.float())
        A, a, H, sub = self.action_embed_dim, self.action_dim, self.horizon, self.sub
        last = (past_rows[:, -1].float() if past_rows is not None
                else z.new_zeros(N, a, dtype=torch.float32))                # (N, a)

        if space == "chunk":
            init = last.repeat(1, sub).reshape(N, 1, A).expand(N, H, A)
            expand = lambda c: c                                             # noqa: E731
        elif space in ("hold", "linear"):
            init = last[:, None].expand(N, H, a)
            if space == "hold":
                expand = lambda c: c.repeat_interleave(sub, dim=1).reshape(c.shape[0], H, A)  # noqa: E731
            else:
                t = (torch.arange(1, sub + 1, device=z.device, dtype=torch.float32) / sub)

                def expand(c):                                               # (M, H, a) knots
                    M = c.shape[0]
                    start = torch.cat([last.repeat_interleave(M // N, 0)[:, None], c[:, :-1]], 1)
                    rows = start[:, :, None] + (c - start)[:, :, None] * t[None, None, :, None]
                    return rows.reshape(M, H, A)
        else:
            raise ValueError(f"unknown plan space '{space}'")

        def cost_fn(cand):
            M = cand.shape[0]
            S = M // N
            full = expand(cand)
            costs = []
            for lo in range(0, M, piece):
                hi = min(M, lo + piece)
                env = torch.arange(lo, hi, device=cand.device) // S
                final = self.rollout_latent(
                    z[env], None if past is None else past[env], full[lo:hi].to(z.dtype)
                )
                costs.append((final.float() - goal[env].float()).abs().mean(dim=(1, 2)))
            return torch.cat(costs)

        plan = cem(cost_fn, init, cfg, generator=generator)
        return expand(plan).reshape(N, self.chunk_len, self.action_dim)
