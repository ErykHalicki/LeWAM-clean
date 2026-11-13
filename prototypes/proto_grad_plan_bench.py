"""Timing for gradient planning: ms per Adam step (WM fwd+bwd) vs the one-time costs.
  python prototypes/proto_grad_plan_bench.py --ckpt ... --roots ... --device cpu
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from probe_suite import Windows, load, log  # noqa: E402

FUT = ["scene_future", "wrist_future", "state_future"]


def resolve_ckpt(spec):
    """A local .pt path, or a wandb reference: `id`, `id:best`, `entity/project/id[:best]`.
    Bare ids default to halicki/lewam-testing."""
    import os
    if Path(spec).exists():
        return spec
    from lewam.utils.ckpt_utils import download_latest_checkpoint, split_alias
    run, alias = split_alias(spec)
    if "/" not in run:
        run = f"halicki/lewam-testing/{run}"
    return download_latest_checkpoint(run, os.path.expanduser("~/.cache/lewam_ckpts"),
                                      alias=alias)

COST = ["scene_future", "wrist_future"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
               help="local .pt, or wandb id (uvqlwtjh / uvqlwtjh:best / entity/project/id)")
    p.add_argument("--roots", default="", help="repo_id=local_root overrides; omit to use the HF cache/hub")
    p.add_argument("--device", default="cpu")
    p.add_argument("--reps", type=int, default=20)
    args = p.parse_args()
    device = torch.device(args.device)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)
    model, cfg, step = load(resolve_ckpt(args.ckpt), device, root_map)
    model.eval()
    win = Windows(cfg.dataset, model.norm_stats, 8, 0, split="val")
    full = next(win.batches(8, device))

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    for B in (1, 8):
        batch = {k: (v[:B] if torch.is_tensor(v) else v) for k, v in full.items()}
        subset = set(model.predictor.required_streams(FUT))
        with torch.no_grad():
            t0 = time.perf_counter()
            clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
            sync(); t_cond = time.perf_counter() - t0
            gt_plan = clean["action_clean"].detach().clone()
            queries = {n: model.predictor.query_embed[n].expand(B, -1, -1) for n in FUT}
            goal = {s: model._encode(s, model.predict_spec[s], batch).float().detach()
                    for s in FUT}
            t0 = time.perf_counter()
            torch.manual_seed(0)
            model.predict(batch, streams=["action"])
            sync(); t_policy = time.perf_counter() - t0

        def wm(a):
            toks = {**clean, "action_clean": a, **queries}
            out = model.predictor(toks, None, sources, cond=cond)
            return sum(((out[s][:, -1] - goal[s][:, -1]) ** 2).mean(-1) for s in COST)

        with torch.no_grad():
            for _ in range(3):
                wm(gt_plan)
            t0 = time.perf_counter()
            for _ in range(args.reps):
                wm(gt_plan)
            sync(); t_fwd = (time.perf_counter() - t0) / args.reps

        a = gt_plan.clone().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=0.05)
        for _ in range(3):
            opt.zero_grad(); wm(a).sum().backward(); opt.step()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            opt.zero_grad(); wm(a).sum().backward(); opt.step()
        sync(); t_step = (time.perf_counter() - t0) / args.reps

        log(f"B={B}  encoder(_condition) {t_cond*1e3:8.1f} ms once | "
            f"policy flow sample {t_policy*1e3:7.1f} ms once | "
            f"WM fwd {t_fwd*1e3:6.1f} ms | Adam fwd+bwd step {t_step*1e3:6.1f} ms | "
            f"50 steps = {t_step*50:5.2f}s")


if __name__ == "__main__":
    main()
