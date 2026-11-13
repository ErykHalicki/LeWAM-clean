"""Best-of-K ranking with GAUSSIAN-NOISED candidates around the policy sample a0, instead of
K independent flow-noise draws. proto_progress_k_variance.py found the flow-sample spread is
tiny (pairwise dist^2 ~0.027 in action space, ~0.0014 relative in predicted latent space) --
this checks whether widening the candidate pool with explicit noise gives the surrogate a real
distribution to rank, sweeping noise_std to see if/where ranking signal appears as spread grows.

Candidates: a0 + noise_std * expert_std * randn_like(a0), K per state per noise_std. Still
on/near-manifold perturbations of a real policy sample (not a gradient-found adversarial
point) -- closer in spirit to the plan's ORIGINAL Gaussian-candidate ranking probe (weak Ridge
head, rho ~0-0.07), but with the strong concat MLP head.

  python prototypes/proto_progress_rank_k_noise.py --ckpt <run_id> --device cuda \
    --n-states 32 --K 16 --noise-stds 0.05,0.1,0.2,0.4 \
    --out prototypes/out/rank_k_noise.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from probe_suite import Windows, load, log  # noqa: E402
from proto_progress_plan import ACT_CLEAN, FUT, resolve_ckpt, wm_scene_future  # noqa: E402
from proto_progress_concat_head import build_concat_head, wm_wrist_future  # noqa: E402
from proto_progress_plan_concat import progress_of_concat  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", default="")
    p.add_argument("--n-states", type=int, default=32)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--noise-stds", default="0.05,0.1,0.2,0.4")
    p.add_argument("--enc-episodes", type=int, default=100)
    p.add_argument("--enc-frames", type=int, default=100)
    p.add_argument("--ks", default="0,1,2")
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/rank_k_noise.json")
    args = p.parse_args()
    device = torch.device(args.device)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)
    ks = [int(x) for x in args.ks.split(",")]
    noise_stds = [float(x) for x in args.noise_stds.split(",")]

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

    results = dict(ckpt=args.ckpt, step=step, head=head_metrics, K=args.K,
                   noise_stds=noise_stds, ks=ks, states=[])
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
            expert_std = expert.std()

        for nstd in noise_stds:
            with torch.no_grad():
                noise = torch.randn(B, args.K, *a0.shape[1:], device=device)
                cand_stack = a0.unsqueeze(1) + nstd * expert_std * noise    # (B, K, 36, 7)

                mse = ((cand_stack - expert.unsqueeze(1)) ** 2).mean(dim=(-1, -2))  # (B, K)
                a_flat = cand_stack.reshape(B, args.K, -1)
                pd2 = ((a_flat[:, :, None, :] - a_flat[:, None, :, :]) ** 2).sum(-1)  # (B,K,K)
                eye = torch.eye(args.K, dtype=torch.bool, device=device)
                a_pairwise = pd2[:, ~eye].mean(-1)                          # (B,)

                surr = torch.zeros(B, args.K, len(ks), device=device)
                for j in range(args.K):
                    a = cand_stack[:, j]
                    sf = wm_scene_future(model, *ctx, a)
                    wf = wm_wrist_future(model, *ctx, a)
                    prog = progress_of_concat(mlp, mlp_mu, mlp_sd, sf, wf)   # (B, 3)
                    surr[:, j] = prog

            for i in range(B):
                mse_i = mse[i].cpu().numpy()
                for ki, k in enumerate(ks):
                    surr_i = surr[i, :, ki].cpu().numpy()
                    rho, _ = spearmanr(surr_i, -mse_i)
                    top1 = int(surr_i.argmax())
                    best = int(mse_i.argmin())
                    results["states"].append(dict(
                        state=seen + i, k=k, noise_std=nstd,
                        action_pairwise_dist2=float(a_pairwise[i]),
                        spearman=float(rho) if not np.isnan(rho) else None,
                        mse_top1=float(mse_i[top1]),
                        mse_best=float(mse_i[best]),
                        mse_random=float(mse_i[0]),
                        mse_mean=float(mse_i.mean()),
                        top1_is_best=bool(top1 == best),
                    ))
        seen += B
        log(f"  {seen}/{args.n_states} states done")
        Path(args.out).write_text(json.dumps(results, indent=1))
        if seen >= args.n_states:
            break

    log("\n" + "=" * 100)
    log(f"{args.ckpt} step {step}   concat-head test R2 {head_metrics['test_r2']:+.3f}"
        f"   K={args.K}   {seen} states")
    log("Gaussian-noised best-of-K: does ranking signal appear as candidate spread widens?")
    log("=" * 100)
    S = results["states"]
    for nstd in noise_stds:
        pd2 = np.mean([r["action_pairwise_dist2"] for r in S if r["noise_std"] == nstd])
        log(f"  noise_std={nstd}  (achieved pairwise action dist^2 = {pd2:.4f})")
        for k in ks:
            rows = [r for r in S if r["k"] == k and r["noise_std"] == nstd]
            rhos = [r["spearman"] for r in rows if r["spearman"] is not None]
            top1 = np.mean([r["mse_top1"] for r in rows])
            best = np.mean([r["mse_best"] for r in rows])
            rand = np.mean([r["mse_random"] for r in rows])
            mean_ = np.mean([r["mse_mean"] for r in rows])
            hit = np.mean([r["top1_is_best"] for r in rows])
            log(f"    k={k}: spearman {np.mean(rhos):+.3f} (median {np.median(rhos):+.3f})  "
                f"top1-is-best {hit:.2f}  mse: top1 {top1:.4f} / best {best:.4f} / "
                f"random {rand:.4f} / mean {mean_:.4f}")

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
