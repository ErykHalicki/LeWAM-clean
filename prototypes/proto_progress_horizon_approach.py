"""Sanity check on the WM's raw predictive quality, independent of any progress-MLP surrogate:
does the WM's own multi-step scene_future prediction monotonically APPROACH the true frame at
the farthest offset as k increases?

z_goal = true encoded frame at the farthest raw_index offset (+36 frames, k=2's own target).
sf = wm_scene_future(model, ctx, a)  ->  (B, 3, 512) at k=0,1,2 (+12/+24/+36 frames).
Hypothesis: ||sf[:,0] - z_goal|| > ||sf[:,1] - z_goal|| > ||sf[:,2] - z_goal||  (k=2 should be
~0 when conditioned on the TRUE action, since that IS its supervised target).

Reports this under BOTH conditioning actions:
  - the TRUE (expert) action chunk -- k=2 should trivially be near-zero (it's the direct
    training target); mainly a sanity check that the pipeline is wired correctly.
  - the POLICY's own sampled action chunk -- the planning-relevant version: as the WM looks
    farther out under the policy's own guess, does its prediction get closer to where the
    demonstrator actually ended up?

  python prototypes/proto_progress_horizon_approach.py --ckpt <run_id> --device cuda \
    --n-states 40 --out prototypes/out/horizon_approach.json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", default="")
    p.add_argument("--n-states", type=int, default=40)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/horizon_approach.json")
    args = p.parse_args()
    device = torch.device(args.device)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)

    model, cfg, step = load(resolve_ckpt(args.ckpt), device, root_map)
    model.eval()
    model.num_flow_steps = args.flow_steps
    for prm in model.parameters():
        prm.requires_grad_(False)
    log(f"loaded step={step}")
    assert "scene_future" in model.predict_spec

    win = Windows(cfg.dataset, model.norm_stats, args.n_states, seed=7, split="val")
    subset = set(model.predictor.required_streams(FUT))

    results = dict(ckpt=args.ckpt, step=step, states=[])
    seen = 0
    for batch in win.batches(args.bs, device):
        with torch.no_grad():
            clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
            B = clean[ACT_CLEAN].shape[0]
            queries = {n: model.predictor.query_embed[n].expand(B, -1, -1) for n in FUT}
            ctx = (clean, sources, cond, queries)

            # true encoded frames at the model's own raw offsets: (B, 3, 512), k=0,1,2
            goal_all = model._encode("scene_future", model.predict_spec["scene_future"], batch)
            z_goal = goal_all[:, 2].float()                  # farthest offset, +36 frames

            expert = clean[ACT_CLEAN].detach().clone()
            torch.manual_seed(0)
            a0 = model.predict(batch, streams=["action"])["action"].detach()

            sf_expert = wm_scene_future(model, *ctx, expert)  # (B, 3, 512)
            sf_policy = wm_scene_future(model, *ctx, a0)

            def dists(sf):
                return [(sf[:, k].float() - z_goal).pow(2).sum(-1).sqrt() for k in range(3)]

            d_expert = dists(sf_expert)                       # list of 3 (B,) tensors
            d_policy = dists(sf_policy)

        for i in range(B):
            results["states"].append(dict(
                state=seen + i,
                d_expert=[float(d[i]) for d in d_expert],
                d_policy=[float(d[i]) for d in d_policy],
                z_goal=z_goal[i].cpu().tolist(),
            ))
        seen += B
        log(f"  {seen}/{args.n_states} states done")
        Path(args.out).write_text(json.dumps(results, indent=1))
        if seen >= args.n_states:
            break

    log("\n" + "=" * 90)
    log(f"{args.ckpt} step {step}   {seen} states")
    log("Does the WM's predicted scene_future monotonically approach the true +36f frame as k -> 2?")
    log("=" * 90)
    S = results["states"]
    z_all = np.array([r["z_goal"] for r in S])                # (N, 512), true encoded goals
    n = len(z_all)
    rng = np.random.default_rng(0)
    pairs_i = np.arange(n)
    pairs_j = (pairs_i + 1 + rng.integers(0, n - 1, size=n)) % n   # never self-paired
    cross = np.sqrt(((z_all[pairs_i] - z_all[pairs_j]) ** 2).sum(-1)).mean()
    log(f"  cross-window scale (different-state reference, {n} pairs): {cross:.3f}")
    for tag in ["d_expert", "d_policy"]:
        arr = np.array([r[tag] for r in S])                  # (N, 3)
        means = arr.mean(0)
        monotone = np.mean(np.all(np.diff(-arr, axis=1) > 0, axis=1))  # dist strictly decreasing
        log(f"  {tag}: k=0 {means[0]:.3f} -> k=1 {means[1]:.3f} -> k=2 {means[2]:.3f}   "
            f"(frac strictly decreasing per-state: {monotone:.2f})")

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
