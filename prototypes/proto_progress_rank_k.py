"""Best-of-K ranking, as an alternative to gradient ascent: draw K INDEPENDENT policy action
chunks (different flow-noise seeds, no perturbation/optimization) per held-out state, score
each with the concat progress MLP on [scene_future; wrist_future], and check whether the
surrogate's top pick is closer to the expert than a random pick or the pool average.

Every candidate is a genuine policy sample, so there is no gradient into the WM and no way to
leave the data manifold -- this sidesteps the Goodhart failure mode entirely by construction.
The remaining question: among K valid samples, does the surrogate's ranking correlate with
which one is actually closer to the expert?

Cf. the plan's original ranking probe (Gaussian / policy-perturbation / mixed candidate pools,
weak Ridge head, Spearman rho ~0-0.07) -- this differs on both axes: candidates are independent
POLICY samples (not perturbations of one), and the head is the strong concat MLP (R2 ~0.92 on
true latents), not Ridge.

  python prototypes/proto_progress_rank_k.py --ckpt <run_id> --device cuda \
    --n-states 32 --K 16 --out prototypes/out/rank_k.json
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
    p.add_argument("--enc-episodes", type=int, default=100)
    p.add_argument("--enc-frames", type=int, default=100)
    p.add_argument("--ks", default="0,1,2")
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/rank_k.json")
    args = p.parse_args()
    device = torch.device(args.device)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)
    ks = [int(x) for x in args.ks.split(",")]

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

    results = dict(ckpt=args.ckpt, step=step, head=head_metrics, K=args.K, ks=ks, states=[])
    seen = 0
    for batch in win.batches(args.bs, device):
        with torch.no_grad():
            clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
            B = clean[ACT_CLEAN].shape[0]
            queries = {n: model.predictor.query_embed[n].expand(B, -1, -1) for n in FUT}
            ctx = (clean, sources, cond, queries)
            expert = clean[ACT_CLEAN].detach().clone()

            # K independent policy samples: no manual_seed between draws -> live RNG state
            cands = [model.predict(batch, streams=["action"])["action"].detach()
                     for _ in range(args.K)]           # each (B, 36, 7)
            cand_stack = torch.stack(cands, dim=1)      # (B, K, 36, 7)

            mse = ((cand_stack - expert.unsqueeze(1)) ** 2).mean(dim=(-1, -2))  # (B, K)

            surr = torch.zeros(B, args.K, len(ks), device=device)
            for j, a in enumerate(cands):
                sf = wm_scene_future(model, *ctx, a)
                wf = wm_wrist_future(model, *ctx, a)
                prog = progress_of_concat(mlp, mlp_mu, mlp_sd, sf, wf)   # (B, 3)
                surr[:, j] = prog

        for i in range(B):
            mse_i = mse[i].cpu().numpy()               # (K,)
            for ki, k in enumerate(ks):
                surr_i = surr[i, :, ki].cpu().numpy()   # (K,)
                rho, _ = spearmanr(surr_i, -mse_i)      # high surrogate should mean low mse
                top1 = int(surr_i.argmax())
                best = int(mse_i.argmin())
                results["states"].append(dict(
                    state=seen + i, k=k,
                    spearman=float(rho) if not np.isnan(rho) else None,
                    mse_top1=float(mse_i[top1]),
                    mse_best=float(mse_i[best]),
                    mse_random=float(mse_i[0]),          # first draw, arbitrary but fixed
                    mse_mean=float(mse_i.mean()),
                    top1_is_best=bool(top1 == best),
                ))
        seen += B
        log(f"  {seen}/{args.n_states} states done")
        Path(args.out).write_text(json.dumps(results, indent=1))
        if seen >= args.n_states:
            break

    log("\n" + "=" * 96)
    log(f"{args.ckpt} step {step}   concat-head test R2 {head_metrics['test_r2']:+.3f}"
        f"   K={args.K}   {seen} states")
    log("best-of-K ranking: does picking the surrogate's top candidate beat random / mean?")
    log("=" * 96)
    S = results["states"]
    for k in ks:
        rows = [r for r in S if r["k"] == k]
        rhos = [r["spearman"] for r in rows if r["spearman"] is not None]
        top1 = np.mean([r["mse_top1"] for r in rows])
        best = np.mean([r["mse_best"] for r in rows])
        rand = np.mean([r["mse_random"] for r in rows])
        mean_ = np.mean([r["mse_mean"] for r in rows])
        hit = np.mean([r["top1_is_best"] for r in rows])
        log(f"  k={k}: spearman rho {np.mean(rhos):+.3f} (median {np.median(rhos):+.3f})  "
            f"top1-is-best {hit:.2f}  mse: top1 {top1:.4f} / oracle-best {best:.4f} / "
            f"random {rand:.4f} / pool-mean {mean_:.4f}")

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
