"""One-shot prototype: gradient planning through a selfwam MSE world model.

Given a holdout window's own contexts, treat the 36-action chunk fed to the world
model (the `action_clean` stream) as the plan variable and Adam it through the
frozen predictor to minimize the distance between the predicted future latents and
the GT future latents (the "goal" = the window's real +1.2s frames, encoded).

Reports, per init (policy sample / persistence / noise) and mode (free / clip3):
goal-cost before/after, chunk MSE vs the GT actions before/after (does refinement
move TOWARD the demonstrated actions or off-manifold?), drift, max|a|.

  python prototypes/proto_grad_plan.py \
    --ckpt <path/to/ckpt.pt> \
    --roots ehalicki/b601_pusht_pick_and_place=<datasets_root>/b601_pusht_pick_and_place_gop1_256/ \
    --n 8 --steps 50 --lr 0.05 --device cpu
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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


def wm_pred(model, clean, sources, cond, queries, a):
    toks = {**clean, "action_clean": a, **queries}
    out = model.predictor(toks, None, sources, cond=cond)
    return {n: out[n] for n in FUT}


def goal_cost(pred, goal):
    """Final-step visual latent distance, per window (B,)."""
    return sum(((pred[s][:, -1] - goal[s][:, -1]) ** 2).mean(-1) for s in COST)


def descend(model, clean, sources, cond, queries, goal, a0, steps, lr, clip=None):
    a = a0.clone().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    curve = []
    for i in range(steps):
        c = goal_cost(wm_pred(model, clean, sources, cond, queries, a), goal)
        curve.append(c.mean().item())
        opt.zero_grad()
        c.sum().backward()
        opt.step()
        if clip is not None:
            with torch.no_grad():
                a.clamp_(-clip, clip)
    return a.detach(), curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
               help="local .pt, or wandb id (uvqlwtjh / uvqlwtjh:best / entity/project/id)")
    p.add_argument("--roots", default="", help="repo_id=local_root overrides; omit to use the HF cache/hub")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="/tmp/grad_plan_proto.json")
    args = p.parse_args()
    device = torch.device(args.device)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)

    model, cfg, step = load(resolve_ckpt(args.ckpt), device, root_map)
    model.eval()
    log(f"loaded step={step}; predict streams: {list(model.predict_spec)}")
    win = Windows(cfg.dataset, model.norm_stats, args.n, 0, split="val")
    batch = next(win.batches(args.n, device))

    with torch.no_grad():
        subset = set(model.predictor.required_streams(FUT))
        clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
        assert "action_clean" in clean, f"clean streams: {list(clean)}"
        gt_plan = clean["action_clean"].detach().clone()
        assert gt_plan.shape[1:] == (36, 7), gt_plan.shape
        queries = {n: model.predictor.query_embed[n].expand(gt_plan.shape[0], -1, -1)
                   for n in FUT}
        goal = {s: model._encode(s, model.predict_spec[s], batch).float().detach()
                for s in FUT}
        # scale of the latent space: cross-window GT distance at the final step
        cross = {s: ((goal[s][:, -1] - goal[s].roll(1, 0)[:, -1]) ** 2).mean().item()
                 for s in COST}
        torch.manual_seed(0)
        policy = model.predict(batch, streams=["action"])["action"].detach()
        assert policy.shape == gt_plan.shape, policy.shape
        hist = hidden["action_context"] if "action_context" in hidden else None
        pers = (hist[:, -1:].expand_as(gt_plan).contiguous() if hist is not None
                else gt_plan[:, :1].expand_as(gt_plan).contiguous())
        noise = torch.randn_like(gt_plan) * gt_plan.std()

    def report(tag, a0, a1, curve):
        with torch.no_grad():
            c0 = goal_cost(wm_pred(model, clean, sources, cond, queries, a0), goal)
            c1 = goal_cost(wm_pred(model, clean, sources, cond, queries, a1), goal)
            m0 = F.mse_loss(a0, gt_plan).item()
            m1 = F.mse_loss(a1, gt_plan).item()
            drift = (a1 - a0).pow(2).mean().sqrt().item()
        rec = dict(tag=tag, cost0=c0.mean().item(), cost1=c1.mean().item(),
                   mse_gt0=m0, mse_gt1=m1, drift=drift,
                   maxabs=a1.abs().max().item(), curve0=curve[0], curveN=curve[-1])
        log(f"{tag:18s} cost {rec['cost0']:.4f}->{rec['cost1']:.4f}  "
            f"mse_vs_gt {m0:.4f}->{m1:.4f}  drift {drift:.3f}  max|a| {rec['maxabs']:.2f}")
        return rec

    with torch.no_grad():
        c_gt = goal_cost(wm_pred(model, clean, sources, cond, queries, gt_plan), goal)
    log(f"latent scale: cross-window {cross}; cost at GT chunk (model floor) "
        f"{c_gt.mean().item():.4f}")

    results = dict(step=step, n=args.n, lr=args.lr, steps=args.steps,
                   cross=cross, cost_gt=c_gt.mean().item(), runs=[])
    for name, init in [("policy", policy), ("persistence", pers), ("noise", noise)]:
        for mode, clip in [("free", None), ("clip3", 3.0)]:
            a1, curve = descend(model, clean, sources, cond, queries, goal,
                                init, args.steps, args.lr, clip=clip)
            results["runs"].append(report(f"{name}/{mode}", init, a1, curve))
    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
