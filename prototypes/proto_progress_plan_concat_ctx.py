"""Same as proto_progress_plan_concat.py, plus a context-anchored latent-drift penalty:
mu * (||scene_future(a)[:,k] - scene_context[:,-1]||^2 + ||wrist_future(a)[:,k] - wrist_context[:,-1]||^2)

scene_context[:,-1] / wrist_context[:,-1] are TRUE encoded frames (the most recent real
observation), not a frozen copy of the policy's own prediction -- unlike the existing
lam*||a-a0||^2 trust region (which anchors the ACTION to a0, itself potentially already
somewhat off-manifold), this anchors the PREDICTED LATENT to something real. Motivation:
concat_head fixed the surrogate's ranking on true latents but left the full optimization
sweep Goodharting just as hard (frac dmse<0 = 0.00, latent norm ratio 0.69-0.77) -- the
mis-ranking and the exploitability are separate problems, and the exploit is specifically
the predicted latent leaving the region anything real lives in.

  python prototypes/proto_progress_plan_concat_ctx.py --ckpt <run_id> --device cuda \
    --n-states 15 --steps 50 --lams 0 --mus 0,0.1,1.0,10.0,100.0 \
    --out prototypes/out/progress_plan_ctx.json
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
from proto_progress_concat_head import build_concat_head, wm_wrist_future  # noqa: E402
from proto_progress_plan_concat import progress_of_concat  # noqa: E402


def optimize(model, ctx, mlp, mu_sd, z_now, a0, k, lam, mu, steps, lr):
    clean, sources, cond, queries = ctx
    z_scene_now, z_wrist_now = z_now
    mlp_mu, mlp_sd = mu_sd
    a = a0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    curve = []
    for _ in range(steps):
        sf = wm_scene_future(model, clean, sources, cond, queries, a)
        wf = wm_wrist_future(model, clean, sources, cond, queries, a)
        prog = progress_of_concat(mlp, mlp_mu, mlp_sd, sf, wf)   # (B, 3)
        surr = prog[:, k]
        trust = ((a - a0) ** 2).mean(dim=(-1, -2))
        ctx_drift = (((sf[:, k] - z_scene_now) ** 2).mean(-1)
                     + ((wf[:, k] - z_wrist_now) ** 2).mean(-1))
        prior = (a ** 2).mean(dim=(-1, -2))
        loss = (-surr + lam * trust + mu * ctx_drift + 0.01 * prior).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(surr.detach().mean()))
    return a.detach(), curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", default="")
    p.add_argument("--n-states", type=int, default=15)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--enc-episodes", type=int, default=100)
    p.add_argument("--enc-frames", type=int, default=100)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--ks", default="0,1,2")
    p.add_argument("--lams", default="0")
    p.add_argument("--mus", default="0,0.1,1.0,10.0,100.0")
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/progress_plan_ctx.json")
    args = p.parse_args()
    device = torch.device(args.device)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)
    ks = [int(x) for x in args.ks.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    mus = [float(x) for x in args.mus.split(",")]

    model, cfg, step = load(resolve_ckpt(args.ckpt), device, root_map)
    model.eval()
    model.num_flow_steps = args.flow_steps
    for prm in model.parameters():
        prm.requires_grad_(False)
    log(f"loaded step={step}")
    assert "scene_future" in model.predict_spec and "wrist_future" in model.predict_spec

    mlp, mlp_mu, mlp_sd, head_metrics = build_concat_head(
        model, cfg, device, args.enc_episodes, args.enc_frames)

    win = Windows(cfg.dataset, model.norm_stats, args.n_states, seed=7, split="val")
    subset = set(model.predictor.required_streams(FUT))

    results = dict(ckpt=args.ckpt, step=step, head=head_metrics,
                   steps=args.steps, lr=args.lr, ks=ks, lams=lams, mus=mus, states=[])
    seen = 0
    for batch in win.batches(args.bs, device):
        with torch.no_grad():
            clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
            assert "scene_context" in clean and "wrist_context" in clean, list(clean)
            B = clean[ACT_CLEAN].shape[0]
            queries = {n: model.predictor.query_embed[n].expand(B, -1, -1) for n in FUT}
            ctx = (clean, sources, cond, queries)
            z_now = (clean["scene_context"][:, -1].float(), clean["wrist_context"][:, -1].float())

            expert = clean[ACT_CLEAN].detach().clone()
            torch.manual_seed(0)
            a0 = model.predict(batch, streams=["action"])["action"].detach()
            assert a0.shape == expert.shape, (a0.shape, expert.shape)

            sf0 = wm_scene_future(model, *ctx, a0)
            wf0 = wm_wrist_future(model, *ctx, a0)
            sfE = wm_scene_future(model, *ctx, expert)
            wfE = wm_wrist_future(model, *ctx, expert)
            prog0 = progress_of_concat(mlp, mlp_mu, mlp_sd, sf0, wf0)
            progE = progress_of_concat(mlp, mlp_mu, mlp_sd, sfE, wfE)
            n0 = torch.cat([sf0, wf0], dim=-1).norm(dim=-1).mean(-1)

        for k in ks:
            for lam in lams:
                for mu in mus:
                    a1, curve = optimize(model, ctx, mlp, (mlp_mu, mlp_sd), z_now, a0,
                                         k, lam, mu, args.steps, args.lr)
                    with torch.no_grad():
                        sf1 = wm_scene_future(model, *ctx, a1)
                        wf1 = wm_wrist_future(model, *ctx, a1)
                        prog1 = progress_of_concat(mlp, mlp_mu, mlp_sd, sf1, wf1)
                        n1 = torch.cat([sf1, wf1], dim=-1).norm(dim=-1).mean(-1)
                        mse_start = ((a0 - expert) ** 2).mean(dim=(-1, -2))
                        mse_final = ((a1 - expert) ** 2).mean(dim=(-1, -2))
                        drift = ((a1 - a0) ** 2).mean(dim=(-1, -2))
                    for i in range(B):
                        results["states"].append(dict(
                            state=seen + i, k=k, lam=lam, mu=mu,
                            mse_start=float(mse_start[i]), mse_final=float(mse_final[i]),
                            dmse=float(mse_final[i] - mse_start[i]),
                            surr_start=float(prog0[i, k]), surr_final=float(prog1[i, k]),
                            surr_expert=float(progE[i, k]),
                            drift=float(drift[i]),
                            norm_ratio=float(n1[i] / n0[i]),
                            curve_first=curve[0], curve_last=curve[-1],
                        ))
        seen += B
        log(f"  {seen}/{args.n_states} states done")
        Path(args.out).write_text(json.dumps(results, indent=1))
        if seen >= args.n_states:
            break

    log("\n" + "=" * 100)
    log(f"{args.ckpt} step {step}   concat-head test R2 {head_metrics['test_r2']:+.3f}"
        f"   {seen} states")
    log("mu = weight on ||predicted latent - last real context latent||^2 (scene+wrist)")
    log("=" * 100)
    log(f"  {'k':>2} {'lam':>5} {'mu':>7} {'mean dmse':>10} {'frac dmse<0':>12} "
        f"{'surr start->final':>19} {'surr expert':>12} {'mean drift':>11} {'||sf|| ratio':>12}")
    S = results["states"]
    for k in ks:
        for lam in lams:
            for mu in mus:
                rows = [r for r in S if r["k"] == k and r["lam"] == lam and r["mu"] == mu]
                dm = np.array([r["dmse"] for r in rows])
                ss = np.mean([r["surr_start"] for r in rows])
                sf = np.mean([r["surr_final"] for r in rows])
                se = np.mean([r["surr_expert"] for r in rows])
                dr = np.mean([r["drift"] for r in rows])
                nr = np.mean([r["norm_ratio"] for r in rows])
                log(f"  {k:>2} {lam:>5.2g} {mu:>7.3g} {dm.mean():>+10.4f} {np.mean(dm < 0):>12.2f} "
                    f"  {ss:>+7.3f} -> {sf:>+7.3f}   {se:>+12.3f} {dr:>11.4f} {nr:>12.2f}")

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
