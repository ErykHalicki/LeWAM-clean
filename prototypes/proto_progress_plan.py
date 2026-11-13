"""Prototype: gradient-optimize an action chunk to maximize a learned task-progress
surrogate scored on a world model's predicted future latents.

Pipeline per held-out state:
  1. train (once per ckpt) a progress MLP:  proj_scene frame latent -> frame_index/(L-1)
     on 100 episodes x 100 frames, episode-disjoint 55/15/30 train/val/test.
  2. start from the policy's own sampled 36-action chunk a0.
  3. Adam on the chunk `a` (fed as the `action_clean` stream) through the FROZEN world
     model -> scene_future (B, 3, 512) at +12/+24/+36 frames -> progress MLP -> progress.
  4. maximize progress at target step k in {0,1,2} SEPARATELY (does the nearer, more
     accurate WM prediction give a more useful gradient than the farther one?), under a
     trust-region penalty lam*||a-a0||^2 for lam in {0, 0.1, 1.0}.

Reports per (k, lam): action-MSE-to-expert before vs after (negative delta = the surrogate
moved the chunk toward the demonstrator; positive delta while surrogate rises = Goodhart),
surrogate start/final/expert, drift, predicted-latent-norm ratio (blow-up = off-distribution).

  python prototypes/proto_progress_plan.py \
    --ckpt <run_id> \
    --n-states 40 --enc-episodes 100 --enc-frames 100 \
    --steps 120 --lr 0.05 --device cuda --out /tmp/progress_plan.json

Notes / gotchas (verified on the selfwam_mot_mse arm):
  - scene_future / state_future / wrist_future are `type: mse` (query-token regression,
    one trunk forward, NOT flow) -> differentiable, cheap, no ODE.
  - the action-conditioning stream the WM reads is `action_clean` (index 0..35), NOT
    `action_context` (via: null, dead). action_entry()/rollout_* drive the wrong stream;
    write `clean["action_clean"]` (== batch["action"][:, 36:72]) directly.
  - model._condition / model.predictor(...) are grad-capable; model.predict() is not.
  - the attends graph couples all 3 future streams: request all query_names, read
    scene_future.
  - scene_future is in proj_scene space (encoder: proj_scene), the same space the MLP
    is trained on -- via `model._encode("scene_future", ...)` for targets and
    `model.run_encoder("proj_scene", frames)` for the MLP training set.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from probe_suite import Windows, load, log  # noqa: E402

FUT = ["scene_future", "wrist_future", "state_future"]
ACT_CLEAN = "action_clean"
SCENE_FIELD = "observation.images.zed_left"


# --------------------------------------------------------------- ckpt resolution
def resolve_ckpt(spec):
    import os
    if Path(spec).exists():
        return spec
    from lewam.utils.ckpt_utils import download_latest_checkpoint, split_alias
    run, alias = split_alias(spec)
    if "/" not in run:
        run = f"halicki/lewam-testing/{run}"
    return download_latest_checkpoint(run, os.path.expanduser("~/.cache/lewam_ckpts"),
                                      alias=alias)


# --------------------------------------------------------------- progress MLP
class MLP(nn.Module):
    def __init__(self, d_in, h=256, layers=3, p=0.15):
        super().__init__()
        mods = [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p)]
        for _ in range(layers - 1):
            mods += [nn.Linear(h, h), nn.ReLU(), nn.Dropout(p)]
        mods.append(nn.Linear(h, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def r2(pred, true):
    return 1.0 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)


@torch.no_grad()
def encode_proj_scene(model, frames_u8, device, bs=8):
    """frames_u8: (N, H, W, C) uint8 -> (N, 512) proj_scene latents."""
    out = []
    for i in range(0, len(frames_u8), bs):
        chunk = frames_u8[i:i + bs].to(device).permute(0, 3, 1, 2).unsqueeze(1).float()
        z = model.run_encoder("proj_scene", chunk)          # (b, 1, 512)
        out.append(z.reshape(z.shape[0], -1).float().cpu())
    return torch.cat(out)


def build_progress_head(model, cfg, device, n_ep, n_frames, seed=0):
    """Encode n_ep x n_frames proj_scene latents, train an MLP progress head, freeze it.
    Returns (mlp, mu, sd, metrics)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    repo_id = str(cfg.dataset.repo_id)
    ds = LeRobotDataset(repo_id)
    lengths = list(ds.meta.episodes["length"])
    starts = list(ds.meta.episodes["dataset_from_index"])
    rng = np.random.default_rng(seed)
    eps = sorted(rng.choice(len(lengths), size=min(n_ep, len(lengths)),
                            replace=False).tolist())
    log(f"progress head: encoding {len(eps)} episodes x ~{n_frames} frames from {repo_id}")

    def load_u8(gidx):
        img = ds[gidx][SCENE_FIELD]                          # (3, H, W) float 0..1
        return (img * 255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()

    Z, y, epv = [], [], []
    for ei, ep in enumerate(eps):
        st, L = int(starts[ep]), int(lengths[ep])
        local = np.unique(np.linspace(0, L - 1, n_frames).round().astype(int))
        raw = torch.stack([load_u8(st + int(t)) for t in local])
        Z.append(encode_proj_scene(model, raw, device))
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
    log(f"progress head: val R2 {best:+.3f}  test R2 {te_r2:+.3f}  ({len(y)} frames)")
    return net, torch.tensor(mu, dtype=torch.float32, device=device), \
        torch.tensor(sd, dtype=torch.float32, device=device), metrics


# --------------------------------------------------------------- differentiable surrogate
def wm_scene_future(model, clean, sources, cond, queries, a):
    toks = {**clean, ACT_CLEAN: a, **queries}
    out = model.predictor(toks, None, sources, cond=cond)
    return out["scene_future"]                               # (B, 3, 512)


def progress_of(mlp, mu, sd, z):                             # z: (..., 512)
    return mlp((z - mu) / sd)                                # (...,)


def optimize(model, ctx, mlp, mu, sd, a0, k, lam, steps, lr):
    """Adam on the chunk to maximize progress_of(scene_future[:, k]). Returns a_final, curve."""
    clean, sources, cond, queries = ctx
    a = a0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    curve = []
    for _ in range(steps):
        sf = wm_scene_future(model, clean, sources, cond, queries, a)
        prog = progress_of(mlp, mu, sd, sf)                  # (B, 3)
        surr = prog[:, k]
        trust = ((a - a0) ** 2).mean(dim=(-1, -2))
        prior = (a ** 2).mean(dim=(-1, -2))
        loss = (-surr + lam * trust + 0.01 * prior).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        curve.append(float(surr.mean()))
    return a.detach(), curve


# --------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="local .pt or wandb run id")
    p.add_argument("--roots", default="", help="repo_id=local_root overrides, comma-sep")
    p.add_argument("--n-states", type=int, default=40)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--enc-episodes", type=int, default=100)
    p.add_argument("--enc-frames", type=int, default=100)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--ks", default="0,1,2", help="target future steps (+12/+24/+36 frames)")
    p.add_argument("--lams", default="0,0.1,1.0", help="trust-region weights")
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/progress_plan.json")
    args = p.parse_args()
    device = torch.device(args.device)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)
    ks = [int(x) for x in args.ks.split(",")]
    lams = [float(x) for x in args.lams.split(",")]

    model, cfg, step = load(resolve_ckpt(args.ckpt), device, root_map)
    model.eval()
    model.num_flow_steps = args.flow_steps
    for prm in model.parameters():
        prm.requires_grad_(False)
    log(f"loaded step={step}; predict streams: {list(model.predict_spec)}")
    assert "scene_future" in model.predict_spec, "not a *_wam arm"

    mlp, mu, sd, head_metrics = build_progress_head(
        model, cfg, device, args.enc_episodes, args.enc_frames)

    win = Windows(cfg.dataset, model.norm_stats, args.n_states, seed=7, split="val")
    subset = set(model.predictor.required_streams(FUT))

    results = dict(ckpt=args.ckpt, step=step, head=head_metrics,
                   steps=args.steps, lr=args.lr, ks=ks, lams=lams, states=[])
    seen = 0
    for batch in win.batches(args.bs, device):
        with torch.no_grad():
            clean, sources, cond, hidden = model._condition(batch, token_streams=subset)
            assert ACT_CLEAN in clean, f"clean streams: {list(clean)}"
            B = clean[ACT_CLEAN].shape[0]
            queries = {n: model.predictor.query_embed[n].expand(B, -1, -1) for n in FUT}
            ctx = (clean, sources, cond, queries)

            expert = clean[ACT_CLEAN].detach().clone()       # (B, 36, 7) normalized
            torch.manual_seed(0)
            a0 = model.predict(batch, streams=["action"])["action"].detach()
            assert a0.shape == expert.shape, (a0.shape, expert.shape)

            sf0 = wm_scene_future(model, *ctx, a0)
            sfE = wm_scene_future(model, *ctx, expert)
            prog0 = progress_of(mlp, mu, sd, sf0)            # (B, 3)
            progE = progress_of(mlp, mu, sd, sfE)
            n0 = sf0.norm(dim=-1).mean(-1)                   # (B,)

        for k in ks:
            for lam in lams:
                a1, curve = optimize(model, ctx, mlp, mu, sd, a0, k, lam,
                                     args.steps, args.lr)
                with torch.no_grad():
                    sf1 = wm_scene_future(model, *ctx, a1)
                    prog1 = progress_of(mlp, mu, sd, sf1)
                    n1 = sf1.norm(dim=-1).mean(-1)
                    mse_start = ((a0 - expert) ** 2).mean(dim=(-1, -2))
                    mse_final = ((a1 - expert) ** 2).mean(dim=(-1, -2))
                    drift = ((a1 - a0) ** 2).mean(dim=(-1, -2))
                for i in range(B):
                    results["states"].append(dict(
                        state=seen + i, k=k, lam=lam,
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

    # ---- summary table
    log("\n" + "=" * 96)
    log(f"{args.ckpt} step {step}   progress head test R2 {head_metrics['test_r2']:+.3f}"
        f"   {seen} states")
    log("k = target future step (0/1/2 = +12/+24/+36 frames);  dmse<0 = optimization moved "
        "chunk TOWARD expert")
    log("=" * 96)
    log(f"  {'k':>2} {'lam':>5} {'mean dmse':>10} {'frac dmse<0':>12} "
        f"{'surr start->final':>19} {'surr expert':>12} {'mean drift':>11} {'||sf|| ratio':>12}")
    S = results["states"]
    for k in ks:
        for lam in lams:
            rows = [r for r in S if r["k"] == k and r["lam"] == lam]
            dm = np.array([r["dmse"] for r in rows])
            ss = np.mean([r["surr_start"] for r in rows])
            sf = np.mean([r["surr_final"] for r in rows])
            se = np.mean([r["surr_expert"] for r in rows])
            dr = np.mean([r["drift"] for r in rows])
            nr = np.mean([r["norm_ratio"] for r in rows])
            log(f"  {k:>2} {lam:>5.2g} {dm.mean():>+10.4f} {np.mean(dm < 0):>12.2f} "
                f"  {ss:>+7.3f} -> {sf:>+7.3f}   {se:>+12.3f} {dr:>11.4f} {nr:>12.2f}")

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
