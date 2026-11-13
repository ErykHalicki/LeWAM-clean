"""Offline probe suite over trained checkpoints: FK error of the predicted chunk at scale,
attentive-pool attention maps, latent variance, and action decodability from pooled vs
raw grid features (is one query a bottleneck?).

    python prototypes/probe_suite.py --ckpts <list file> --out /tmp/probe_suite \
        --roots ehalicki/b601_pusht_pick_and_place=<datasets_root>/b601_pusht_pick_and_place_gop1_256/,ehalicki/b601_pusht=<datasets_root>/b601_pusht_gop1_256/

<list file>: one "<name> <ckpt path>" per line. Each checkpoint rebuilds from its own
embedded config (exact as-trained key order), so renamed arms are not an issue. Held-out
windows are shared across models by global frame, per dataset. Results go to
<out>/<name>.json (incremental; existing files are skipped) and <out>/<name>_attn.png.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lewam.debug.chunk_fk import _fk_positions  # noqa: E402
from lewam.utils.spec import parse_index, raw_index, spec_fields  # noqa: E402


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def fk_err(pred_deg, truth_deg):
    return np.linalg.norm(_fk_positions(pred_deg) - _fk_positions(truth_deg), axis=-1)


# ---------------------------------------------------------------- data

class Windows:
    """Held-out windows of one dataset family, decoded once and kept in RAM. Picks are
    global frames, so two spec families of the same dataset score the same moments."""

    def __init__(self, dcfg, stats, n, seed, frames=None, workers=8, split="val"):
        from torch.utils.data import DataLoader, Subset

        from lewam.datasets import build_dataset

        self.ds = build_dataset(dcfg, norm_stats_override=stats, augment=False, split=split)
        src = self.ds
        while not hasattr(src, "_index") and hasattr(src, "dataset"):
            src = src.dataset
        by_frame = {entry[0]: i for i, entry in enumerate(src._index)}
        if frames is None:
            rng = np.random.default_rng(seed)
            all_frames = np.array(sorted(by_frame))
            frames = rng.choice(all_frames, size=min(n, len(all_frames)), replace=False)
        self.frames = [int(f) for f in frames if int(f) in by_frame]
        idx = [by_frame[f] for f in self.frames]
        self.episodes = [int(src._index[i][2]) for i in idx]
        loader = DataLoader(Subset(self.ds, idx), batch_size=None, num_workers=workers)
        self.items = list(loader)
        log(f"decoded {len(self.items)} windows ({len(self.ds)} in split)")

    def batches(self, bs, device, n=None):
        items = self.items if n is None else self.items[:n]
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            yield {k: torch.stack([it[k] for it in chunk]).to(device)
                   for k in chunk[0] if torch.is_tensor(chunk[0][k])}


# ---------------------------------------------------------------- model

def load(ckpt, device, root_map):
    from lewam.algorithms import build_algorithm
    from lewam.utils.checkpoint import load_checkpoint
    from lewam.utils.ckpt_utils import config_from_checkpoint

    stored = config_from_checkpoint(ckpt)
    if stored is None:
        raise RuntimeError(f"{ckpt} has no embedded config")
    dcfg = stored.dataset
    root = root_map.get(str(dcfg.get("repo_id")))
    if root:
        dcfg.root = root
    elif dcfg.get("root") and not Path(str(dcfg.root)).exists():
        # the checkpoint's stored root is a devbox path; with no override, let the
        # lerobot backend resolve the repo_id from the HF cache (or download it)
        dcfg.root = None
    model = build_algorithm(stored.algorithm)
    step = load_checkpoint(model, None, ckpt)
    return model.to(device).eval(), stored, step


def spec_signature(cfg):
    from omegaconf import OmegaConf
    return json.dumps(OmegaConf.to_container(cfg.dataset, resolve=True), sort_keys=True)


# ---------------------------------------------------------------- probes

def fk_probe(model, win, device, bs=16):
    from lewam.utils.normalize import Normalizer
    stats = model.norm_stats
    normalizer = Normalizer(stats, method=getattr(model, "norm_method", "mean_std")).to(device)
    field = model.action_field
    rows = parse_index(raw_index(model.predict_spec[model.action_stream]))
    start = rows.index(0) if 0 in rows else 0
    per = []
    for obs in win.batches(bs, device):
        with torch.no_grad():
            o = model.predict(obs)
        pred = o[model.action_stream] if isinstance(o, dict) else o
        window = obs[field][:, -len(rows):]
        truth = window[:, start:start + pred.shape[1]]
        mse = ((pred - truth) ** 2).mean(dim=(1, 2)).cpu().numpy()
        pred_deg = normalizer.unnormalize(field, pred).cpu().numpy()
        truth_deg = normalizer.unnormalize(field, truth).cpu().numpy()
        for b in range(len(pred_deg)):
            e = fk_err(pred_deg[b], truth_deg[b])
            hold = fk_err(np.repeat(truth_deg[b][:1], len(truth_deg[b]), 0), truth_deg[b])
            dv_p, dv_t = np.diff(pred_deg[b], axis=0), np.diff(truth_deg[b], axis=0)
            per.append({
                "mean": float(e.mean()), "final": float(e[-1]), "hold_mean": float(hold.mean()),
                "mse_norm": float(mse[b]),
                "step_delta_rms_deg": float(np.sqrt((dv_p ** 2).mean())),
                "truth_delta_rms_deg": float(np.sqrt((dv_t ** 2).mean())),
                "jerk_rms_deg": float(np.sqrt((np.diff(dv_p, axis=0) ** 2).mean())),
                "truth_jerk_rms_deg": float(np.sqrt((np.diff(dv_t, axis=0) ** 2).mean())),
            })
    return summarize_fk(per)


def summarize_fk(per):
    m = np.array([w["mean"] for w in per]) * 100
    f = np.array([w["final"] for w in per]) * 100
    h = np.array([w["hold_mean"] for w in per]) * 100
    out = {
        "n": len(per), "mean_cm": float(m.mean()), "median_cm": float(np.median(m)),
        "p90_cm": float(np.percentile(m, 90)), "final_mean_cm": float(f.mean()),
        "hold_mean_cm": float(h.mean()), "hold_median_cm": float(np.median(h)),
        "mse_norm": float(np.mean([w["mse_norm"] for w in per])),
    }
    for k in ("step_delta_rms_deg", "truth_delta_rms_deg", "jerk_rms_deg", "truth_jerk_rms_deg"):
        if k in per[0]:
            out[k] = float(np.mean([w[k] for w in per]))
    out["per_window_mean_cm"] = [round(float(x), 3) for x in m]
    return out


def fk_probe_ac(model, win, device, n=64, bs=16, plan=None):
    """VJEPA2-AC: CEM plan toward the sample's own +1.2 s frame, scored like a chunk."""
    from lewam.utils.normalize import Normalizer
    plan = plan or {"space": "linear", "init_std": 0.1, "num_samples": 128, "n_steps": 30,
                    "topk": 16, "chunk": 512}
    normalizer = Normalizer(model.norm_stats, method=model.norm_method).to(device)
    field = model.action_field
    per, tf, ar = [], [], []
    for obs in win.batches(bs, device, n=n):
        with torch.no_grad():
            out = model.loss(obs)
            tf.append(float(out["loss/tf"])); ar.append(float(out["loss/ar"]))
            frames = model._raw(model.ctx_stream, model.conditioning[model.ctx_stream], obs)
            future = model._raw(model.future_stream, model.predict_spec[model.future_stream], obs)
            rows = model._raw(model.action_cond, model.conditioning[model.action_cond], obs).float()
            past_n = model.past_steps * model.sub
            past, gt = rows[:, :past_n], rows[:, past_n:]
            goal = model.encode_frames(future[:, -1:])[:, 0]
            pred = model.plan(frames, goal, past if past_n else None, plan,
                              generator=torch.Generator(device=device).manual_seed(0))
        mse = ((pred - gt) ** 2).mean(dim=(1, 2)).cpu().numpy()
        pred_deg = normalizer.unnormalize(field, pred).cpu().numpy()
        truth_deg = normalizer.unnormalize(field, gt).cpu().numpy()
        for b in range(len(pred_deg)):
            e = fk_err(pred_deg[b], truth_deg[b])
            hold = fk_err(np.repeat(truth_deg[b][:1], len(truth_deg[b]), 0), truth_deg[b])
            per.append({"mean": float(e.mean()), "final": float(e[-1]),
                        "hold_mean": float(hold.mean()), "mse_norm": float(mse[b])})
    s = summarize_fk(per)
    s.update({"loss_tf": float(np.mean(tf)), "loss_ar": float(np.mean(ar)), "plan": plan})
    return s


def variance_probe(model, win, device, batches=4, bs=16):
    tot, n = {}, 0
    for i, obs in enumerate(win.batches(bs, device)):
        if i >= batches:
            break
        with torch.no_grad():
            out = model.loss(obs)
        for k, v in out.items():
            if k == "loss" or k.startswith(("loss/", "var/")):
                tot[k] = tot.get(k, 0.0) + float(v)
        n += 1
    return {k: v / n for k, v in tot.items()}


def pool_modules(model):
    """{stream: (pool encoder name, root encoder name, camera field, raw stream spec)}."""
    out = {}
    specs = getattr(model, "encoder_specs", {})
    for name, spec in model.conditioning.items():
        enc = spec.get("encoder")
        if enc is None:
            continue
        chain = model._encoder_chains[enc]
        pools = [e for e in chain if specs[e].get("type") == "attentive_pool"]
        if pools:
            out[name] = (pools[0], chain[0], spec_fields(name, spec)[0], spec)
    return out


def pool_attention(pool, grid_tokens):
    """Replicates AttentivePoolEncoder's cross-attention softmax.
    grid_tokens (B, T*P, in_dim) -> (B, T, Q, P) weights averaged over heads."""
    b, n, _ = grid_tokens.shape
    span = pool.tokens_per_step
    src = pool.norm_src(grid_tokens.reshape(b * (n // span), span, -1))
    attn = pool.attn
    H = attn.num_heads

    def heads(x):
        return x.unflatten(-1, (H, x.shape[-1] // H)).transpose(1, 2)

    q = attn.q_norm(heads(attn.q_proj(pool.query.expand(src.shape[0], -1, -1))))
    k = attn.k_norms[0](heads(attn.k_projs[0](src)))
    w = F.softmax(q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1)
    return w.mean(1).reshape(b, n // span, pool.num_queries, span)


def attention_probe(model, win, device, out_png, bs=16, n_png=4):
    """Per camera: attention entropy (fraction of uniform), max weight, spatial spread,
    correlation with frame-difference motion energy; overlays for the first windows."""
    res, panels = {}, {}
    for stream, (pool_name, root, field, spec) in pool_modules(model).items():
        pool = model.encoders[pool_name]
        P = pool.tokens_per_step
        g = int(round(math.sqrt(P)))
        ent, mx, spread, corr, first = [], [], [], [], None
        for obs in win.batches(bs, device):
            raw = model._raw(stream, spec, obs)                       # (B, T, C, H, W) uint8
            with torch.no_grad():
                grid = model.run_encoder(root, raw)                   # (B, T*P, D)
                w = pool_attention(pool, grid)                        # (B, T, Q, P)
            wq = w.mean(2)                                            # over queries
            e = -(wq * (wq + 1e-12).log()).sum(-1) / math.log(P)
            ent.append(e.flatten().cpu()); mx.append(wq.max(-1).values.flatten().cpu())
            ys, xs = torch.meshgrid(torch.arange(g, device=device), torch.arange(g, device=device), indexing="ij")
            coords = torch.stack([ys.flatten(), xs.flatten()], -1).float() / (g - 1)
            mean_xy = wq @ coords                                     # (B, T, 2)
            var = (wq[..., None] * (coords[None, None] - mean_xy[:, :, None]) ** 2).sum((-1, -2))
            spread.append(var.sqrt().flatten().cpu())
            if raw.shape[1] >= 2:
                f = raw.float().mean(2)                               # (B, T, H, W) grey
                motion = (f[:, 1:] - f[:, :-1]).abs()
                motion = F.adaptive_avg_pool2d(motion, g).flatten(2)  # (B, T-1, P)
                a = wq[:, 1:]
                ac = a - a.mean(-1, keepdim=True); mc = motion - motion.mean(-1, keepdim=True)
                c = (ac * mc).sum(-1) / (ac.norm(dim=-1) * mc.norm(dim=-1) + 1e-8)
                corr.append(c.flatten().cpu())
            if first is None:
                first = (raw[:n_png, -1].cpu(), w[:n_png, -1].cpu())
        res[stream] = {
            "num_queries": int(pool.num_queries), "tokens": int(P),
            "entropy_frac": float(torch.cat(ent).mean()), "max_weight": float(torch.cat(mx).mean()),
            "spread": float(torch.cat(spread).mean()),
            "motion_corr": float(torch.cat(corr).mean()) if corr else None,
        }
        panels[stream] = first
    if panels:
        try:
            render_attention(panels, out_png)
        except Exception as err:  # noqa: BLE001
            log("attention png failed:", err)
    return res


def render_attention(panels, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    streams = list(panels)
    n = panels[streams[0]][0].shape[0]
    fig, axes = plt.subplots(len(streams), n, figsize=(3 * n, 3 * len(streams)), squeeze=False)
    for r, s in enumerate(streams):
        frames, w = panels[s]
        for c in range(n):
            img = frames[c].permute(1, 2, 0).numpy()
            a = w[c].mean(0)                                           # (P,) over queries
            g = int(round(math.sqrt(a.numel())))
            heat = F.interpolate(a.view(1, 1, g, g), size=img.shape[:2], mode="bilinear")[0, 0]
            ax = axes[r, c]; ax.imshow(img); ax.imshow(heat.numpy(), cmap="jet", alpha=0.45)
            ax.set_axis_off(); ax.set_title(f"{s} q={w.shape[1]}", fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=80); plt.close(fig)


def ridge_r2(X, Y, groups=None, lams=(1e-2, 1e-1, 1, 10, 100, 1000), train=0.75, seed=0):
    """Closed-form ridge from features X (N, F) to targets Y (N, K); best-lambda test R^2
    against the train-mean predictor. `groups` (episode per row) keeps whole episodes on
    one side of the split, so neighbouring windows cannot leak across it."""
    N = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    if groups is None:
        perm = torch.randperm(N, generator=g)
        ntr = int(N * train)
        tr, te = perm[:ntr], perm[ntr:]
    else:
        groups = torch.as_tensor(groups)
        uniq = groups.unique()
        uniq = uniq[torch.randperm(len(uniq), generator=g)]
        n_tr_ep = max(1, int(len(uniq) * train))
        tr_eps = uniq[:n_tr_ep]
        mask = torch.isin(groups, tr_eps)
        tr, te = mask.nonzero().flatten(), (~mask).nonzero().flatten()
        ntr = len(tr)
    Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - Xm) / Xs
    Ym = Y[tr].mean(0)
    Xtr, Ytr, Xte, Yte = Xn[tr].double(), (Y[tr] - Ym).double(), Xn[te].double(), (Y[te] - Ym).double()
    G = Xtr.T @ Xtr
    best = None
    for lam in lams:
        W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), Xtr.T @ Ytr)
        r2 = 1 - ((Xte @ W - Yte) ** 2).sum() / (Yte ** 2).sum()
        if best is None or r2 > best[0]:
            best = (float(r2), lam)
    return {"r2": best[0], "lam": best[1], "n_train": int(ntr), "n_test": int(N - ntr), "dim": int(X.shape[1])}


def decodability_probe(model, win, device, bs=16, proj_dim=1024, seed=0):
    """Can the GT action chunk be linearly decoded from (a) the pooled/projected latent
    the trunk actually sees, (b) the frame-mean of the raw ViT grid, (c) a random
    projection of the full grid? (b)/(c) >> (a) means the pool is throwing away
    action-relevant information."""
    if model.action_stream is None:
        return {}
    pools = pool_modules(model)
    if not pools:
        return {}
    rows = parse_index(raw_index(model.predict_spec[model.action_stream]))
    start = rows.index(0) if 0 in rows else 0
    feats = {"pooled": [], "grid_mean": [], "grid_proj": []}
    Y = []
    R = None
    for obs in win.batches(bs, device):
        chunk = obs[model.action_field][:, -len(rows):][:, start:]
        Y.append(chunk.flatten(1).float().cpu())
        p_, m_, r_ = [], [], []
        for stream, (pool_name, root, field, spec) in pools.items():
            raw = model._raw(stream, spec, obs)
            with torch.no_grad():
                grid = model.run_encoder(root, raw)                  # (B, T*P, D)
                pooled = model.run_encoder(spec["encoder"], raw)     # (B, T*Q, d) trunk input
            B, TP, D = grid.shape
            P = model.encoders[pool_name].tokens_per_step
            gridf = grid.view(B, TP // P, P, D)
            m_.append(gridf.mean(2).flatten(1))
            if R is None:
                R = torch.randn(P * D, proj_dim, generator=torch.Generator().manual_seed(seed)) / math.sqrt(P * D)
                R = R.to(device)
            r_.append((gridf.flatten(2) @ R).flatten(1))
            p_.append(pooled.flatten(1))
        feats["pooled"].append(torch.cat(p_, 1).float().cpu())
        feats["grid_mean"].append(torch.cat(m_, 1).float().cpu())
        feats["grid_proj"].append(torch.cat(r_, 1).float().cpu())
    Y = torch.cat(Y)
    out = {}
    groups = win.episodes[: Y.shape[0]]
    for k, v in feats.items():
        X = torch.cat(v)
        out[k] = ridge_r2(X.to(device), Y.to(device), groups=groups)
        out[k]["split"] = "by_episode"
    out["n_episodes"] = int(len(set(groups)))
    return out


def sensitivity_probe(model, win, device, bs=16, n=256):
    """How much each conditioning field matters: the chunk error when that field comes
    from a DIFFERENT window (batch rolled by one) while everything else stays. A field
    the model ignores costs nothing; one it relies on costs a lot. Reported as the
    normalized MSE and FK with the swap, against the intact prediction."""
    if model.action_stream is None:
        return {}
    from lewam.utils.normalize import Normalizer
    normalizer = Normalizer(model.norm_stats, method=getattr(model, "norm_method", "mean_std")).to(device)
    field = model.action_field
    rows = parse_index(raw_index(model.predict_spec[model.action_stream]))
    start = rows.index(0) if 0 in rows else 0
    cond_fields = sorted({f for name, spec in model.conditioning.items() for f in spec_fields(name, spec)})
    variants = {"intact": None, **{f: f for f in cond_fields}}
    acc = {k: {"mse": [], "fk": []} for k in variants}
    for obs in win.batches(bs, device, n=n):
        window = obs[field][:, -len(rows):]
        for k, swap in variants.items():
            o2 = dict(obs)
            if swap is not None:
                if swap == field:
                    # only the HISTORY rows are conditioning; the chunk stays the window's own
                    a = obs[field].clone()
                    hist = a.shape[1] - len(rows) + start
                    a[:, :hist] = torch.roll(a[:, :hist], 1, dims=0)
                    o2[field] = a
                else:
                    o2[swap] = torch.roll(obs[swap], 1, dims=0)
            with torch.no_grad():
                o = model.predict(o2)
            pred = o[model.action_stream] if isinstance(o, dict) else o
            truth = window[:, start:start + pred.shape[1]]
            acc[k]["mse"].append(((pred - truth) ** 2).mean(dim=(1, 2)).cpu())
            pd = normalizer.unnormalize(field, pred).cpu().numpy()
            td = normalizer.unnormalize(field, truth).cpu().numpy()
            acc[k]["fk"].extend(fk_err(pd[b], td[b]).mean() * 100 for b in range(len(pd)))
    out = {}
    base_mse = float(torch.cat(acc["intact"]["mse"]).mean())
    base_fk = float(np.mean(acc["intact"]["fk"]))
    for k, v in acc.items():
        mse = float(torch.cat(v["mse"]).mean()); fk = float(np.mean(v["fk"]))
        out[k] = {"mse": mse, "fk_cm": fk, "mse_ratio": mse / base_mse, "fk_delta_cm": fk - base_fk}
    return out


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", required=True)
    p.add_argument("--roots", default="", help="repo_id=root[,repo_id=root]")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="/tmp/probe_suite")
    p.add_argument("--skip-done", action="store_true", default=True)
    p.add_argument("--sensitivity", action="store_true", help="also run the input-swap sensitivity probe")
    args = p.parse_args()
    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root_map = dict(kv.split("=", 1) for kv in args.roots.split(",") if kv)
    entries = [ln.split() for ln in Path(args.ckpts).read_text().splitlines() if ln.strip() and not ln.startswith("#")]

    windows = {}          # (repo, signature) -> Windows
    frames_by_repo = {}   # repo -> shared global-frame picks
    for name, ckpt in entries:
        dest = out / f"{name}.json"
        if args.skip_done and dest.exists():
            log(f"skip {name} (done)"); continue
        t0 = time.time()
        try:
            model, cfg, step = load(ckpt, device, root_map)
        except Exception as err:  # noqa: BLE001
            log(f"{name}: LOAD FAILED: {err}"); continue
        repo = str(cfg.dataset.get("repo_id"))
        key = (repo, spec_signature(cfg))
        if key not in windows:
            windows[key] = Windows(cfg.dataset, model.norm_stats, args.n, args.seed,
                                   frames=frames_by_repo.get(repo))
            frames_by_repo.setdefault(repo, windows[key].frames)
        win = windows[key]
        rec = {"name": name, "ckpt": ckpt, "step": step, "algorithm": cfg.algorithm.name, "arm": str(cfg.get("name")),
               "repo_id": repo, "n_windows": len(win.items),
               "params": int(sum(p.numel() for p in model.parameters()))}
        try:
            if cfg.algorithm.name == "vjepa2_ac":
                rec["fk"] = fk_probe_ac(model, win, device)
            elif model.action_stream is not None:
                rec["fk"] = fk_probe(model, win, device)
            rec["variance"] = variance_probe(model, win, device)
            if cfg.algorithm.name == "generic_vit_predictor":
                rec["attention"] = attention_probe(model, win, device, str(out / f"{name}_attn.png"))
                rec["decodability"] = decodability_probe(model, win, device)
            if args.sensitivity and model.action_stream is not None:
                rec["sensitivity"] = sensitivity_probe(model, win, device)
        except Exception as err:  # noqa: BLE001
            import traceback; traceback.print_exc()
            rec["error"] = repr(err)
        rec["seconds"] = round(time.time() - t0, 1)
        dest.write_text(json.dumps(rec, indent=1))
        fk = rec.get("fk") or {}
        log(f"{name}: step {step} fk mean {fk.get('mean_cm', float('nan')):.2f} cm (hold {fk.get('hold_mean_cm', float('nan')):.2f}) "
            f"mse {fk.get('mse_norm', float('nan')):.4f} | {rec.get('attention')} | {rec.get('decodability')} | {rec['seconds']}s")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
