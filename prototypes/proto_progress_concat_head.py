"""Check: does concatenating scene+wrist frame latents give the progress MLP a better /
better-ranked signal, before touching the gradient optimizer at all?

Encodes the same 100ep x 100frames as proto_progress_plan.py, but from BOTH
observation.images.zed_left (proj_scene) and observation.images.wrist_cam (proj_wrist),
concatenates to a 1024-d vector, trains the same 3x256 MLP, and reports:
  - held-out test R2 (does concat beat the proj_scene-only R2 from proto_progress_plan.py?)
  - the Q4 ranking check: on the SAME held-out val windows used by proto_progress_plan.py,
    does the true expert chunk's [scene_future; wrist_future] concat score higher progress
    than the policy's own sampled chunk? (frac(expert>start), mean(expert-start))

Does NOT run the gradient optimizer -- this only asks whether the surrogate itself is
better/better-ranked on TRUE latents (both at encode time and read off the WM's own
scene_future/wrist_future prediction, no optimization).

  python prototypes/proto_progress_concat_head.py --ckpt <run_id> --device cuda \
    --n-states 40 --out prototypes/out/concat_head.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from probe_suite import Windows, log  # noqa: E402
from proto_progress_plan import (  # noqa: E402
    ACT_CLEAN, FUT, MLP, r2, resolve_ckpt, wm_scene_future,
)
from probe_suite import load  # noqa: E402

WRIST_FIELD = "observation.images.wrist_cam"
SCENE_FIELD = "observation.images.zed_left"


@torch.no_grad()
def encode_both(model, scene_u8, wrist_u8, device, bs=8):
    """(N,H,W,C) uint8 x2 -> (N, 1024) concatenated [proj_scene; proj_wrist] latents."""
    out = []
    for i in range(0, len(scene_u8), bs):
        sc = scene_u8[i:i + bs].to(device).permute(0, 3, 1, 2).unsqueeze(1).float()
        wr = wrist_u8[i:i + bs].to(device).permute(0, 3, 1, 2).unsqueeze(1).float()
        zs = model.run_encoder("proj_scene", sc).reshape(sc.shape[0], -1)
        zw = model.run_encoder("proj_wrist", wr).reshape(wr.shape[0], -1)
        out.append(torch.cat([zs, zw], dim=-1).float().cpu())
    return torch.cat(out)


def wm_wrist_future(model, clean, sources, cond, queries, a):
    toks = {**clean, ACT_CLEAN: a, **queries}
    out = model.predictor(toks, None, sources, cond=cond)
    return out["wrist_future"]


def build_concat_head(model, cfg, device, n_ep, n_frames, seed=0):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    repo_id = str(cfg.dataset.repo_id)
    ds = LeRobotDataset(repo_id)
    lengths = list(ds.meta.episodes["length"])
    starts = list(ds.meta.episodes["dataset_from_index"])
    rng = np.random.default_rng(seed)
    eps = sorted(rng.choice(len(lengths), size=min(n_ep, len(lengths)),
                            replace=False).tolist())
    log(f"concat head: encoding {len(eps)} episodes x ~{n_frames} frames from {repo_id}")

    def load_u8(gidx, field):
        img = ds[gidx][field]
        return (img * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()

    Z, y, epv = [], [], []
    for ei, ep in enumerate(eps):
        st, L = int(starts[ep]), int(lengths[ep])
        local = np.unique(np.linspace(0, L - 1, n_frames).round().astype(int))
        scene_raw = torch.stack([load_u8(st + int(t), SCENE_FIELD) for t in local])
        wrist_raw = torch.stack([load_u8(st + int(t), WRIST_FIELD) for t in local])
        Z.append(encode_both(model, scene_raw, wrist_raw, device))
        y.append(torch.tensor(local / (L - 1), dtype=torch.float32))
        epv.append(torch.full((len(local),), ei, dtype=torch.long))
        if (ei + 1) % 20 == 0:
            log(f"  ...{ei + 1}/{len(eps)} episodes")
    Z = torch.cat(Z).numpy().astype(np.float64)
    y = torch.cat(y).numpy().astype(np.float64)
    epv = torch.cat(epv).numpy()
    n_e = int(epv.max()) + 1

    order = np.random.default_rng(seed).permutation(n_e)
    n_te = int(round(n_e * 0.30))
    n_va = max(3, int(round(n_e * 0.15)))
    te = set(order[:n_te].tolist())
    va = set(order[n_te:n_te + n_va].tolist())
    m_tr = ~np.isin(epv, list(te) + list(va))
    m_va = np.isin(epv, list(va))
    m_te = np.isin(epv, list(te))
    mu, sd = Z[m_tr].mean(0), Z[m_tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)

    def std(a):
        return ((a - mu) / sd).astype(np.float32)

    torch.manual_seed(seed)
    net = MLP(Z.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.tensor(std(Z[m_tr]), device=device)
    ytr = torch.tensor(y[m_tr].astype(np.float32), device=device)
    Xva = torch.tensor(std(Z[m_va]), device=device)
    Xte = torch.tensor(std(Z[m_te]), device=device)
    n = len(Xtr)
    best, best_state, bad = -1e9, None, 0
    for epoch in range(400):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            F.mse_loss(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vr2 = r2(net(Xva).cpu().numpy(), y[m_va])
        if vr2 > best:
            best, best_state, bad = vr2, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 40:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        te_r2 = r2(net(Xte).cpu().numpy(), y[m_te])
    for p in net.parameters():
        p.requires_grad_(False)
    metrics = dict(val_r2=float(best), test_r2=float(te_r2),
                   n_samples=int(len(y)), n_episodes=int(n_e))
    log(f"concat head: val R2 {best:+.3f}  test R2 {te_r2:+.3f}  ({len(y)} frames)")
    return net, torch.tensor(mu, dtype=torch.float32, device=device), \
        torch.tensor(sd, dtype=torch.float32, device=device), metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", default="")
    p.add_argument("--n-states", type=int, default=40)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--enc-episodes", type=int, default=100)
    p.add_argument("--enc-frames", type=int, default=100)
    p.add_argument("--ks", default="0,1,2")
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/concat_head.json")
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
    assert "wrist_future" in model.predict_spec, "no wrist_future stream on this arm"

    net, mu, sd, head_metrics = build_concat_head(
        model, cfg, device, args.enc_episodes, args.enc_frames)

    win = Windows(cfg.dataset, model.norm_stats, args.n_states, seed=7, split="val")
    subset = set(model.predictor.required_streams(FUT))

    results = dict(ckpt=args.ckpt, step=step, head=head_metrics, ks=ks, states=[])
    seen = 0
    for batch in win.batches(args.bs, device):
        with torch.no_grad():
            clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
            B = clean[ACT_CLEAN].shape[0]
            queries = {n: model.predictor.query_embed[n].expand(B, -1, -1) for n in FUT}

            expert = clean[ACT_CLEAN].detach().clone()
            torch.manual_seed(0)
            a0 = model.predict(batch, streams=["action"])["action"].detach()

            sf0 = wm_scene_future(model, clean, sources, cond, queries, a0)
            wf0 = wm_wrist_future(model, clean, sources, cond, queries, a0)
            sfE = wm_scene_future(model, clean, sources, cond, queries, expert)
            wfE = wm_wrist_future(model, clean, sources, cond, queries, expert)

            z0 = torch.cat([sf0, wf0], dim=-1)               # (B, 3, 1024)
            zE = torch.cat([sfE, wfE], dim=-1)
            prog0 = net((z0 - mu) / sd)                       # (B, 3)
            progE = net((zE - mu) / sd)

        for k in ks:
            for i in range(B):
                results["states"].append(dict(
                    state=seen + i, k=k,
                    surr_start=float(prog0[i, k]), surr_expert=float(progE[i, k]),
                ))
        seen += B
        log(f"  {seen}/{args.n_states} states done")
        if seen >= args.n_states:
            break

    log("\n" + "=" * 72)
    log(f"{args.ckpt} step {step}   concat-head test R2 {head_metrics['test_r2']:+.3f}"
        f"   {seen} states")
    log("Q4 check: does the TRUE expert chunk score higher progress than the policy sample?")
    log("=" * 72)
    S = results["states"]
    for k in ks:
        rows = [r for r in S if r["k"] == k]
        ss = np.array([r["surr_start"] for r in rows])
        se = np.array([r["surr_expert"] for r in rows])
        log(f"  k={k}: surr_start {ss.mean():+.3f}  surr_expert {se.mean():+.3f}  "
            f"frac(expert>start) {np.mean(se > ss):.2f}  mean(expert-start) {np.mean(se - ss):+.3f}")

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
