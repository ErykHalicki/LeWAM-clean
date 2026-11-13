"""Autoregressive gradient planning: instead of reading the model's own joint 3-step
scene_future head, roll the WM forward 3 TIMES at its nearest offset (+12 frames each), each
time feeding the predicted latent back in as the new scene_context/wrist_context, to reach the
same +36-frame horizon by composition. Optimize the three 12-step action blocks (independent
tensors, one per stage) against the TRUE +36-frame goal latent.

Mechanics: action_clean is a positionally-sensitive 36-slot window (each slot = one raw frame,
individually attended); confirmed empirically that zeroing slots 12:36 changes the +12f (k=0)
prediction negligibly (diff ~0.08 vs the ~15-20 latent scale) while later k's depend on them --
so each stage's forward call places that stage's 12 actions in slots 0:12 and zeros the rest,
reads out ONLY the k=0 (+12f) prediction, and re-conditions scene_context/wrist_context by
sliding the window (old "0" frame -> new "-12" slot, new prediction -> new "0" slot) before the
next stage.

  python prototypes/proto_grad_plan_autoregressive.py --ckpt <run_id> --device cuda \
    --n-states 40 --steps 120 --lr 0.05 --lams 0,0.1,1.0 \
    --out prototypes/out/grad_plan_ar.json
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
from proto_progress_plan import ACT_CLEAN, FUT  # noqa: E402
from proto_progress_plan import resolve_ckpt  # noqa: E402


def ar_step(model, clean_base, sources, cond, queries, action_block, B, act_len=36):
    """One +12f rollout step: action_block (B,12,7) in slots 0:12, rest zeroed. Returns
    (scene_pred, wrist_pred) at k=0 (+12f), each (B, 512)."""
    a_full = action_block.new_zeros(B, act_len, action_block.shape[-1])
    a_full[:, :action_block.shape[1]] = action_block
    toks = {**clean_base, ACT_CLEAN: a_full, **queries}
    out = model.predictor(toks, None, sources, cond=cond)
    return out["scene_future"][:, 0], out["wrist_future"][:, 0]


def slide_context(ctx_2frame, new_frame):
    """(B, 2, 512) context window -> slide in new_frame (B, 512) as the new '0' slot."""
    return torch.cat([ctx_2frame[:, 1:], new_frame.unsqueeze(1)], dim=1)


def rollout_ar(model, clean0, sources, cond, queries, blocks, B):
    """3-stage AR rollout. blocks: list of 3 (B,12,7) action tensors. Returns the final
    (+36f) predicted (scene, wrist) latents and the per-stage predictions."""
    clean = dict(clean0)
    scene_ctx, wrist_ctx = clean["scene_context"], clean["wrist_context"]
    preds = []
    for block in blocks:
        clean["scene_context"], clean["wrist_context"] = scene_ctx, wrist_ctx
        sp, wp = ar_step(model, clean, sources, cond, queries, block, B)
        preds.append((sp, wp))
        scene_ctx = slide_context(scene_ctx, sp)
        wrist_ctx = slide_context(wrist_ctx, wp)
    return preds[-1][0], preds[-1][1], preds


def optimize(model, clean0, sources, cond, queries, goal_scene, goal_wrist, blocks0, lam,
             steps, lr, B):
    blocks = [b.clone().detach().requires_grad_(True) for b in blocks0]
    opt = torch.optim.Adam(blocks, lr=lr)
    curve = []
    for _ in range(steps):
        sf, wf, _ = rollout_ar(model, clean0, sources, cond, queries, blocks, B)
        cost = (((sf - goal_scene) ** 2).mean(-1) + ((wf - goal_wrist) ** 2).mean(-1))
        trust = sum(((b - b0) ** 2).mean(dim=(-1, -2)) for b, b0 in zip(blocks, blocks0))
        loss = (cost + lam * trust).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(cost.detach().mean()))
    return [b.detach() for b in blocks], curve


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
    p.add_argument("--out", default="/tmp/grad_plan_ar.json")
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

            expert = clean[ACT_CLEAN].detach().clone()        # (B, 36, 7)
            expert_blocks = [expert[:, 12 * i:12 * (i + 1)] for i in range(3)]
            torch.manual_seed(0)
            a0 = model.predict(batch, streams=["action"])["action"].detach()
            a0_blocks = [a0[:, 12 * i:12 * (i + 1)] for i in range(3)]

            goal_scene = model._encode(
                "scene_future", model.predict_spec["scene_future"], batch
            )[:, -1].float().detach()
            goal_wrist = model._encode(
                "wrist_future", model.predict_spec["wrist_future"], batch
            )[:, -1].float().detach()

            sf0, wf0, _ = rollout_ar(model, clean, sources, cond, queries, a0_blocks, B)
            cost0 = ((sf0 - goal_scene) ** 2).mean(-1) + ((wf0 - goal_wrist) ** 2).mean(-1)
            sfE, wfE, _ = rollout_ar(model, clean, sources, cond, queries, expert_blocks, B)
            costE = ((sfE - goal_scene) ** 2).mean(-1) + ((wfE - goal_wrist) ** 2).mean(-1)
            n0 = torch.cat([sf0, wf0], dim=-1).norm(dim=-1)

        for lam in lams:
            blocks1, curve = optimize(model, clean, sources, cond, queries, goal_scene,
                                      goal_wrist, a0_blocks, lam, args.steps, args.lr, B)
            with torch.no_grad():
                sf1, wf1, _ = rollout_ar(model, clean, sources, cond, queries, blocks1, B)
                cost1 = ((sf1 - goal_scene) ** 2).mean(-1) + ((wf1 - goal_wrist) ** 2).mean(-1)
                n1 = torch.cat([sf1, wf1], dim=-1).norm(dim=-1)
                a1 = torch.cat(blocks1, dim=1)
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
    log(f"{args.ckpt} step {step}   {seen} states   (autoregressive 3x +12f rollout, GT +36f goal)")
    log("dmse<0 = optimization moved chunk TOWARD expert; cost = dist to TRUE +36f goal latent")
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
