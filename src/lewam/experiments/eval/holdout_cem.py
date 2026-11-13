"""Held-out loss plus CEM-planned action error for a latent world model.

`OfflineEval` scores `model.loss` -- for a world model that is a latent regression, which
says nothing about the actions the planner would command. This backend additionally plans
on a few held-out samples: context frames in, the goal being the sample's own last future
frame, and scores the planned chunk against the actions the demonstrator executed toward
that frame. `loss/cem_action` is the normalized MSE, the same quantity `loss/int_action`
reports for the flow arms; `loss/cem_action_raw` is in the arm's own units, and
`loss/cem_action_persistence` holds the last executed action for the whole chunk.
"""

import torch
from torch.utils.data import DataLoader

from ...utils.normalize import Normalizer
from .base import EvalResult
from .offline import OfflineEval


class HoldoutCEMEval(OfflineEval):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.plan_cfg = dict(cfg.get("plan") or {})
        self.plan_batches = int(cfg.get("plan_batches", 1))
        self.plan_batch_size = int(cfg.get("plan_batch_size", 16))

    def run(self, model):
        base = super().run(model)
        if not self.plan_batches:
            return base
        net = getattr(model, "module", model)
        for attr in ("plan", "encode_frames", "ctx_stream", "future_stream", "action_cond"):
            if not hasattr(net, attr):
                raise TypeError(
                    f"eval backend holdout_cem plans with a vjepa2_ac-style world model; "
                    f"{type(net).__name__} has no `{attr}`"
                )
        dataset = self._build(model)
        loader = DataLoader(
            dataset, batch_size=self.plan_batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed + 1),
            num_workers=self.num_workers,
        )
        device = next(model.parameters()).device
        stats = getattr(net, "norm_stats", None)
        normalizer = (Normalizer(stats, method=getattr(net, "norm_method", "mean_std"))
                      .to(device) if stats else None)
        totals, seen = {}, 0
        was_training = model.training
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            model.eval()
            try:
                for i, batch in enumerate(loader):
                    if i >= self.plan_batches:
                        break
                    batch = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in batch.items()}
                    out = self._plan_batch(net, batch, normalizer)
                    n = out.pop("n")
                    for key, value in out.items():
                        totals[key] = totals.get(key, 0.0) + value * n
                    seen += n
            finally:
                model.train(was_training)
        metrics = {**base.metrics, **{k: v / seen for k, v in totals.items()},
                   "plan_samples": seen}
        return EvalResult(metrics=metrics, videos=base.videos, images=base.images,
                          episodes=base.episodes)

    @torch.no_grad()
    def _plan_batch(self, net, batch, normalizer):
        frames = net._raw(net.ctx_stream, net.conditioning[net.ctx_stream], batch)
        future = net._raw(net.future_stream, net.predict_spec[net.future_stream], batch)
        rows = net._raw(net.action_cond, net.conditioning[net.action_cond], batch).float()
        past_n = net.past_steps * net.sub
        past, gt = rows[:, :past_n], rows[:, past_n:]
        goal = net.encode_frames(future[:, -1:])[:, 0]
        planned = net.plan(frames, goal, past if past_n else None, self.plan_cfg)
        persist = (past[:, -1:] if past_n else torch.zeros_like(gt[:, :1])).expand_as(gt)
        out = {
            "loss/cem_action": F_mse(planned, gt),
            "loss/cem_action_persistence": F_mse(persist, gt),
        }
        if normalizer is not None:
            field = net.action_field
            out["loss/cem_action_raw"] = F_mse(
                normalizer.unnormalize(field, planned), normalizer.unnormalize(field, gt)
            )
        out["n"] = int(gt.shape[0])
        return out


def F_mse(a, b):
    return ((a - b) ** 2).mean().item()
