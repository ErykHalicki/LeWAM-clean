"""Policy-seeded gradient planning against the TRUE GT future latent (goal image, encoded),
with a trust region -- the oracle-objective sibling of proto_progress_plan.py, on the same
40-state protocol used throughout this investigation (rather than proto_grad_plan.py's
one-shot 8-window batch).

Objective is the GOAL DISTANCE itself (floored at 0, well-posed), not a learned scalar score --
this isolates whether the Goodhart found against the progress-MLP surrogate is a property of
optimizing an UNBOUNDED LEARNED SCORE through the WM, or whether gradient descent through this
WM's predicted latents Goodharts against ANY objective, including a well-posed oracle one.

loss = ||scene_future(a)[:,-1] - goal_scene[:,-1]||^2 + ||wrist_future(a)[:,-1] - goal_wrist[:,-1]||^2
       + lam * ||a - a0||^2
a0 = policy's own sampled chunk (seed). Reports goal-cost and action-MSE-to-expert before/after,
same convention as proto_progress_plan.py's dmse.

  python prototypes/proto_grad_plan_trust.py --ckpt <run_id> --device cuda \
    --n-states 40 --steps 120 --lr 0.05 --lams 0,0.1,1.0 \
    --out prototypes/out/grad_plan_trust.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from probe_suite import Windows, load, log  # noqa: E402
from proto_progress_plan import ACT_CLEAN, FUT, resolve_ckpt, wm_scene_future  # noqa: E402
from proto_progress_concat_head import wm_wrist_future  # noqa: E402


def goal_cost(sf, wf, goal_scene, goal_wrist):
    """Final-step (k=2, +36f) visual latent distance to the true goal, per window (B,)."""
    return (((sf[:, -1] - goal_scene) ** 2).mean(-1)
            + ((wf[:, -1] - goal_wrist) ** 2).mean(-1))


def optimize(model, ctx, goal_scene, goal_wrist, a0, lam, steps, lr):
    clean, sources, cond, queries = ctx
    a = a0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    curve = []
    for _ in range(steps):
        sf = wm_scene_future(model, clean, sources, cond, queries, a)
        wf = wm_wrist_future(model, clean, sources, cond, queries, a)
        cost = goal_cost(sf, wf, goal_scene, goal_wrist)
        trust = ((a - a0) ** 2).mean(dim=(-1, -2))
        loss = (cost + lam * trust).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(cost.detach().mean()))
    return a.detach(), curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", default="")
    p.add_argument("--n-states", type=int, default=40)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lams", default="0,0.1,1.0")
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/grad_plan_trust.json")
    args = p.parse_args()
    device = torch.device(args.device)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)
    lams = [float(x) for x in args.lams.split(",")]

    model, cfg, step = load(resolve_ckpt(args.ckpt), device, root_map)
    model.eval()
    model.num_flow_steps = args.flow_steps
    for prm in model.parameters():
        prm.requires_grad_(False)
    log(f"loaded step={step}")
    assert "scene_future" in model.predict_spec and "wrist_future" in model.predict_spec

    win = Windows(cfg.dataset, model.norm_stats, args.n_states, seed=7, split="val")
    subset = set(model.predictor.required_streams(FUT))

    results = dict(ckpt=args.ckpt, step=step, steps=args.steps, lr=args.lr, lams=lams,
                   states=[])
    seen = 0
    for batch in win.batches(args.bs, device):
        with torch.no_grad():
            clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
            B = clean[ACT_CLEAN].shape[0]
            queries = {n: model.predictor.query_embed[n].expand(B, -1, -1) for n in FUT}
            ctx = (clean, sources, cond, queries)

            expert = clean[ACT_CLEAN].detach().clone()
            torch.manual_seed(0)
            a0 = model.predict(batch, streams=["action"])["action"].detach()

            # true encoded goal at the farthest offset (+36 frames), same slot the
            # optimize() cost reads (k=-1 == k=2)
            goal_scene = model._encode(
                "scene_future", model.predict_spec["scene_future"], batch
            )[:, -1].float().detach()
            goal_wrist = model._encode(
                "wrist_future", model.predict_spec["wrist_future"], batch
            )[:, -1].float().detach()

            sf0 = wm_scene_future(model, *ctx, a0)
            wf0 = wm_wrist_future(model, *ctx, a0)
            cost0 = goal_cost(sf0, wf0, goal_scene, goal_wrist)
            sfE = wm_scene_future(model, *ctx, expert)
            wfE = wm_wrist_future(model, *ctx, expert)
            costE = goal_cost(sfE, wfE, goal_scene, goal_wrist)
            n0 = torch.cat([sf0[:, -1], wf0[:, -1]], dim=-1).norm(dim=-1)

        for lam in lams:
            a1, curve = optimize(model, ctx, goal_scene, goal_wrist, a0, lam,
                                 args.steps, args.lr)
            with torch.no_grad():
                sf1 = wm_scene_future(model, *ctx, a1)
                wf1 = wm_wrist_future(model, *ctx, a1)
                cost1 = goal_cost(sf1, wf1, goal_scene, goal_wrist)
                n1 = torch.cat([sf1[:, -1], wf1[:, -1]], dim=-1).norm(dim=-1)
                mse_start = ((a0 - expert) ** 2).mean(dim=(-1, -2))
                mse_final = ((a1 - expert) ** 2).mean(dim=(-1, -2))
                drift = ((a1 - a0) ** 2).mean(dim=(-1, -2))
            for i in range(B):
                results["states"].append(dict(
                    state=seen + i, lam=lam,
                    mse_start=float(mse_start[i]), mse_final=float(mse_final[i]),
                    dmse=float(mse_final[i] - mse_start[i]),
                    cost_start=float(cost0[i]), cost_final=float(cost1[i]),
                    cost_expert=float(costE[i]),
                    drift=float(drift[i]),
                    norm_ratio=float(n1[i] / n0[i]),
                    curve_first=curve[0], curve_last=curve[-1],
                ))
        seen += B
        log(f"  {seen}/{args.n_states} states done")
        Path(args.out).write_text(json.dumps(results, indent=1))
        if seen >= args.n_states:
            break

    log("\n" + "=" * 96)
    log(f"{args.ckpt} step {step}   {seen} states   (GT goal-image objective, not a learned surrogate)")
    log("dmse<0 = optimization moved chunk TOWARD expert; cost is distance to TRUE goal latent (floored at 0)")
    log("=" * 96)
    log(f"  {'lam':>5} {'mean dmse':>10} {'frac dmse<0':>12} "
        f"{'cost start->final':>19} {'cost expert':>12} {'mean drift':>11} {'||z|| ratio':>12}")
    S = results["states"]
    for lam in lams:
        rows = [r for r in S if r["lam"] == lam]
        dm = np.array([r["dmse"] for r in rows])
        cs = np.mean([r["cost_start"] for r in rows])
        cf = np.mean([r["cost_final"] for r in rows])
        ce = np.mean([r["cost_expert"] for r in rows])
        dr = np.mean([r["drift"] for r in rows])
        nr = np.mean([r["norm_ratio"] for r in rows])
        log(f"  {lam:>5.2g} {dm.mean():>+10.4f} {np.mean(dm < 0):>12.2f} "
            f"  {cs:>7.3f} -> {cf:>7.3f}   {ce:>12.3f} {dr:>11.4f} {nr:>12.2f}")

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
