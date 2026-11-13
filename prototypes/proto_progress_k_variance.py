"""How much do K independent policy samples actually differ, in action space and in the WM's
predicted latent space? If they are nearly identical, proto_progress_rank_k.py's null ranking
result is explained trivially (no real distribution to rank) rather than being a genuine
surrogate failure.

No progress head needed here -- this only measures spread, not whether a surrogate can order
it. Cheap: K forward samples + K trunk forwards per state, no Adam, no MLP training.

  python prototypes/proto_progress_k_variance.py --ckpt <run_id> --device cuda \
    --n-states 32 --K 16 --out prototypes/out/k_variance.json
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


def pairwise_mean_dist(x):
    """x: (K, D) -> mean pairwise squared L2 distance (excluding diagonal)."""
    K = x.shape[0]
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)     # (K, K)
    off = d2[~torch.eye(K, dtype=torch.bool, device=x.device)]
    return off.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", default="")
    p.add_argument("--n-states", type=int, default=32)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--ks", default="0,1,2")
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/k_variance.json")
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

    win = Windows(cfg.dataset, model.norm_stats, args.n_states, seed=7, split="val")
    subset = set(model.predictor.required_streams(FUT))

    results = dict(ckpt=args.ckpt, step=step, K=args.K, ks=ks, states=[])
    seen = 0
    for batch in win.batches(args.bs, device):
        with torch.no_grad():
            clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
            B = clean[ACT_CLEAN].shape[0]
            queries = {n: model.predictor.query_embed[n].expand(B, -1, -1) for n in FUT}
            ctx = (clean, sources, cond, queries)
            expert = clean[ACT_CLEAN].detach().clone()

            cands = [model.predict(batch, streams=["action"])["action"].detach()
                     for _ in range(args.K)]                 # each (B, 36, 7)
            cand_stack = torch.stack(cands, dim=1)            # (B, K, 36, 7)

            # action-space: per-state std across K, and pairwise dist, vs the expert-chunk
            # norm (scale reference) and vs the cross-window (different states) scale.
            a_flat = cand_stack.reshape(B, args.K, -1)        # (B, K, 252)
            expert_flat = expert.reshape(B, -1)               # (B, 252)
            a_std = a_flat.std(dim=1).mean(-1)                # (B,) mean per-dim std across K
            expert_scale = expert_flat.pow(2).mean(-1).sqrt() # (B,) rms scale of the demo chunk
            cross_state = (a_flat[:, 0] - a_flat.roll(1, 0)[:, 0]).pow(2).mean(-1).sqrt()

            sf_list, wf_list = [], []
            for a in cands:
                sf_list.append(wm_scene_future(model, *ctx, a))          # (B, 3, 512)
                wf_list.append(wm_wrist_future(model, *ctx, a))
            sf_stack = torch.stack(sf_list, dim=1)            # (B, K, 3, 512)
            wf_stack = torch.stack(wf_list, dim=1)

        for i in range(B):
            for ki, k in enumerate(ks):
                sf_i = sf_stack[i, :, ki]                     # (K, 512)
                wf_i = wf_stack[i, :, ki]
                a_pd = pairwise_mean_dist(a_flat[i])          # scalar
                sf_pd = pairwise_mean_dist(sf_i)
                wf_pd = pairwise_mean_dist(wf_i)
                sf_norm = sf_i.norm(dim=-1).mean()
                wf_norm = wf_i.norm(dim=-1).mean()
                results["states"].append(dict(
                    state=seen + i, k=k,
                    action_pairwise_dist2=float(a_pd),
                    action_std_per_dim=float(a_std[i]),
                    expert_rms=float(expert_scale[i]),
                    cross_state_dist=float(cross_state[i]),
                    scene_pairwise_dist2=float(sf_pd),
                    scene_norm=float(sf_norm),
                    wrist_pairwise_dist2=float(wf_pd),
                    wrist_norm=float(wf_norm),
                ))
        seen += B
        log(f"  {seen}/{args.n_states} states done")
        Path(args.out).write_text(json.dumps(results, indent=1))
        if seen >= args.n_states:
            break

    log("\n" + "=" * 100)
    log(f"{args.ckpt} step {step}   K={args.K}   {seen} states")
    log("How much do K independent policy samples actually differ?")
    log("=" * 100)
    S = results["states"]
    a_pd = np.mean([r["action_pairwise_dist2"] for r in S])
    a_std = np.mean([r["action_std_per_dim"] for r in S])
    e_rms = np.mean([r["expert_rms"] for r in S])
    cross = np.mean([r["cross_state_dist"] for r in S])
    log(f"  action space: mean pairwise dist^2 across K = {a_pd:.5f}  "
        f"(per-dim std {a_std:.4f}, expert chunk rms {e_rms:.4f}, "
        f"cross-different-state dist {cross:.4f} for scale)")
    for k in ks:
        rows = [r for r in S if r["k"] == k]
        sf_pd = np.mean([r["scene_pairwise_dist2"] for r in rows])
        sf_n = np.mean([r["scene_norm"] for r in rows])
        wf_pd = np.mean([r["wrist_pairwise_dist2"] for r in rows])
        wf_n = np.mean([r["wrist_norm"] for r in rows])
        log(f"  k={k}: scene_future pairwise dist^2 {sf_pd:.4f} (norm {sf_n:.3f}, "
            f"ratio {sf_pd / sf_n**2:.4f})   wrist_future pairwise dist^2 {wf_pd:.4f} "
            f"(norm {wf_n:.3f}, ratio {wf_pd / wf_n**2:.4f})")

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
