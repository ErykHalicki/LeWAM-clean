"""Does a world model's predicted future actually track the future? Latent-space check
of every predicted (non-action) stream against baselines that need no model at all.

A future-latent head can report a respectable loss while predicting nothing: the flow
heads of the first WAM arms sat at the norm of untransported noise, worse than copying
the last context latent. This runs a
checkpoint on HELD-OUT windows and puts its prediction next to the cheapest guesses:

    persist   copy the last context latent forward      (the bar to clear)
    mean      the mean latent of the probe set           (a constant answer)
    cross     the latent of a random OTHER window        (the scale of the space)
    gauss     a random vector with the GT's per-dim moments (an untrained head)
    shuffled  the model's own prediction, driven by ANOTHER window's action chunk
              (dynamics = the answer must change when the actions do)

Everything is a mean L2 to the encoder's own GT latent of that future frame, per future
step, in the model's own latent space (so numbers compare across steps and baselines,
not across arms with different projectors). `cos_delta` is the cosine between the
predicted change (pred - last) and the true change (gt - last): direction, independent
of scale. `action_sens` is how far the prediction moves when only the actions are
swapped, relative to its own error -- near 0 means the head ignores the actions and is
at best a good extrapolator, not a dynamics model.

    python -m lewam.debug.probe_dynamics --run halicki/lewam-testing/asqdowvp:best
    python -m lewam.debug.probe_dynamics --run asqdowvp:best o1mtnxvu:best --n 128
    python -m lewam.debug.probe_dynamics --run /path/to/ckpt.pt --dataset-root ~/data/b601_pusht
    python -m lewam.debug.probe_dynamics --run asqdowvp --from-hub     # no local dataset copy

A run is `<entity>/<project>/<id>[:alias]`, a bare `<id>[:alias]` under --project, or a
local .pt path. Each checkpoint rebuilds from its own embedded config (exact as-trained
key order, see utils/ckpt_utils.py) and its own norm stats; a frozen-encoder arm is
probed with its encoder run live (no encoding cache needed). The windows are picked by
global frame with --seed, so several checkpoints of one dataset score the same moments.
Writes <out>/<name>.json per checkpoint and prints one table per stream. CPU works
(ViT-B, n=64: minutes); `--device auto` takes cuda when there is one.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lewam.utils.spec import parse_index, spec_fields

DEFAULT_PROJECT = "halicki/lewam-testing"


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------------------------------------------------------- model

def resolve_run(ref, project):
    """A CLI checkpoint reference -> (kind, value): ('path', Path) or ('run', run_path)."""
    if ref.endswith(".pt") or Path(ref).is_file():
        return "path", Path(ref)
    if "/" not in ref:
        ref = f"{project}/{ref}"
    return "run", ref


def load_model(ref, device, ckpt_dir, project):
    from lewam.algorithms import build_algorithm
    from lewam.utils.checkpoint import load_checkpoint
    from lewam.utils.ckpt_utils import (config_from_checkpoint, download_latest_checkpoint,
                                        fetch_run_config, split_alias)

    kind, value = resolve_run(ref, project)
    if kind == "path":
        path, run_path = value, None
    else:
        run_path, alias = split_alias(value)
        path = download_latest_checkpoint(run_path, Path(ckpt_dir), alias=alias)
    stored = config_from_checkpoint(path)
    if stored is None:
        if run_path is None:
            raise RuntimeError(f"{path} has no embedded config and no run to fetch it from")
        stored = fetch_run_config(run_path)
    model = build_algorithm(stored.algorithm)
    step = load_checkpoint(model, None, path)
    name = str(stored.get("name") or Path(path).stem)
    return model.to(device).eval(), stored, step, name, path


def future_streams(model):
    """Every predicted stream but the action chunk: the world-model half."""
    return [s for s in model.predict_spec if s != model.action_stream]


def context_of(model, sname):
    """The conditioning stream that reads the same field as predicted stream `sname`,
    i.e. the one whose last latent is the persistence guess."""
    field = spec_fields(sname, model.predict_spec[sname])[0]
    for cname, spec in model.conditioning.items():
        if spec_fields(cname, spec)[0] == field and cname != model.action_stream:
            return cname
    return None


# ---------------------------------------------------------------- data

def unwrap(dataset):
    """The LeRobotSource under the normalize/augment wrappers."""
    src = dataset
    while not hasattr(src, "_index") and hasattr(src, "dataset"):
        src = src.dataset
    return src


def pick_windows(dataset, n, seed):
    """`n` held-out windows drawn by GLOBAL FRAME (so different checkpoints of one dataset
    land on the same moments) -> (dataset indices, episode ordinals)."""
    src = unwrap(dataset)
    by_frame = {entry[0]: i for i, entry in enumerate(src._index)}
    frames = np.array(sorted(by_frame))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(frames, size=min(n, len(frames)), replace=False)
    idx = [by_frame[int(f)] for f in sorted(chosen)]
    return idx, [int(src._index[i][2]) for i in idx]


def decode_windows(dataset, idx, workers):
    from torch.utils.data import DataLoader, Subset
    loader = DataLoader(Subset(dataset, idx), batch_size=None, num_workers=workers)
    return list(loader)


def batches(items, bs, device):
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        yield {k: torch.stack([it[k] for it in chunk]).to(device)
               for k in chunk[0] if torch.is_tensor(chunk[0][k])}


def row_positions(source, field, offsets):
    """Where the rows at relative `offsets` sit inside the batch tensor of `field`: the
    lerobot source serves each field as its sorted union of requested offsets."""
    steps = list(source._windows[field])
    return [steps.index(o) for o in offsets]


# ---------------------------------------------------------------- probe

@torch.no_grad()
def probe(model, items, source, bs, device, seed):
    futures = future_streams(model)
    if not futures:
        raise SystemExit("this checkpoint predicts only the action chunk: nothing to probe")
    # the action rows the futures actually READ (conditioning streams on the action field
    # that some future stream attends; an unattended action history is left alone): those
    # are swapped between windows for `shuffled`
    attended = {a for s in futures for a in model.predict_spec[s].get("attends", [])}
    act_streams = [n for n, s in model.conditioning.items()
                   if spec_fields(n, s)[0] == model.action_field and n in attended]
    act_pos = None
    if act_streams:
        offsets = sorted({o for n in act_streams
                          for o in parse_index(model.conditioning[n]["index"])})
        act_pos = torch.tensor(row_positions(source, model.action_field, offsets))
        log(f"shuffled baseline swaps action rows {offsets[0]}..{offsets[-1]} "
            f"({', '.join(act_streams)}) between windows")
    else:
        log("no future stream attends an action stream: no shuffled baseline")
    acc = {s: {k: [] for k in ("gt", "last", "pred", "pred_sh")} for s in futures}
    gen = torch.Generator().manual_seed(seed)
    for obs in batches(items, bs, device):
        pred = model.predict(obs, streams=futures)
        pred_sh = None
        if act_pos is not None and obs[model.action_field].shape[0] > 1:
            B = obs[model.action_field].shape[0]
            perm = torch.randperm(B, generator=gen)
            while bool((perm == torch.arange(B)).any()):  # every window gets a foreign chunk
                perm = torch.randperm(B, generator=gen)
            swapped = obs[model.action_field].clone()
            swapped[:, act_pos] = obs[model.action_field][perm][:, act_pos]
            pred_sh = model.predict({**obs, model.action_field: swapped}, streams=futures)
        for sname in futures:
            spec = model.predict_spec[sname]
            steps = len(parse_index(spec["index"]))
            if spec.get("encoder"):
                gt = model._encode(sname, spec, obs).float()
                cname = context_of(model, sname)
                last = model._encode(cname, model.conditioning[cname], obs)[:, -1].float()
            else:
                field = spec_fields(sname, spec)[0]
                rows = parse_index(spec.get("raw_index", spec["index"]))
                gt = obs[field][:, row_positions(source, field, rows)].float()
                cname = context_of(model, sname)
                crow = max(parse_index(model.conditioning[cname].get(
                    "raw_index", model.conditioning[cname]["index"])))
                last = obs[field][:, row_positions(source, field, [crow])[0]].float()
            B = gt.shape[0]
            d = acc[sname]
            d["gt"].append(gt.reshape(B, steps, -1).cpu())
            d["last"].append(last.reshape(B, 1, -1).cpu())
            d["pred"].append(pred[sname].float().reshape(B, steps, -1).cpu())
            if pred_sh is not None:
                d["pred_sh"].append(pred_sh[sname].float().reshape(B, steps, -1).cpu())
    return acc


def summarize(acc, seed):
    out = {}
    gen = torch.Generator().manual_seed(seed + 1)
    for sname, d in acc.items():
        gt, last, pred = (torch.cat(d[k]) for k in ("gt", "last", "pred"))
        N, T, D = gt.shape
        mean_lat = gt.mean(0, keepdim=True)
        perm = torch.randperm(N, generator=gen)
        while N > 1 and bool((perm == torch.arange(N)).any()):
            perm = torch.randperm(N, generator=gen)
        gauss = gt.mean(0, keepdim=True) + gt.std(0, keepdim=True) * torch.randn(
            gt.shape, generator=gen)
        l2 = lambda a, b: (a - b).norm(dim=-1).mean(0)  # noqa: E731 - per future step
        st = {
            "l2_pred": l2(pred, gt), "l2_persist": l2(last, gt), "l2_mean": l2(mean_lat, gt),
            "l2_cross": l2(gt[perm], gt), "l2_gauss": l2(gauss, gt),
            "cos_delta": F.cosine_similarity(pred - last, gt - last, dim=-1).mean(0),
        }
        if d["pred_sh"]:
            pred_sh = torch.cat(d["pred_sh"])
            st["l2_shuffled"] = l2(pred_sh, gt)
            st["cos_delta_shuffled"] = F.cosine_similarity(
                pred_sh - last, gt - last, dim=-1).mean(0)
            st["action_sens"] = l2(pred_sh, pred) / st["l2_pred"].clamp_min(1e-12)
        st = {k: [round(float(x), 4) for x in v] for k, v in st.items()}
        st["pred_var_ratio"] = round(float(
            pred.reshape(-1, D).var(0).mean() / gt.reshape(-1, D).var(0).mean().clamp_min(1e-12)), 4)
        st["n"], st["steps"], st["dim"] = N, T, D
        st["verdict"] = verdict(st)
        out[sname] = st
    return out


def verdict(st):
    p, per, mean = np.array(st["l2_pred"]), np.array(st["l2_persist"]), np.array(st["l2_mean"])
    beats = [i + 1 for i in range(len(p)) if p[i] < 0.95 * per[i]]
    ties = [i + 1 for i in range(len(p)) if 0.95 * per[i] <= p[i] <= 1.05 * per[i]]
    parts = []
    if beats:
        parts.append(f"beats persistence at step{'s' if len(beats) > 1 else ''} "
                     f"{_span(beats)} ({(per[[i - 1 for i in beats]] / p[[i - 1 for i in beats]]).max():.2f}x best)")
    if ties:
        parts.append(f"ties persistence at {_span(ties)}")
    worse = [i + 1 for i in range(len(p)) if p[i] > 1.05 * per[i]]
    if worse:
        parts.append(f"WORSE than persistence at {_span(worse)}")
    if (p > mean).all():
        parts.append("worse than the constant mean latent everywhere: head has not learned")
    if "action_sens" in st:
        s = float(np.mean(st["action_sens"]))
        parts.append("ignores actions" if s < 0.05 else
                     "weakly action-conditioned" if s < 0.25 else "action-conditioned")
    return "; ".join(parts)


def _span(ix):
    return f"{ix[0]}..{ix[-1]}" if len(ix) > 2 and ix == list(range(ix[0], ix[-1] + 1)) \
        else ",".join(map(str, ix))


def print_table(name, step, results):
    rows = ["l2_pred", "l2_persist", "l2_mean", "l2_cross", "l2_gauss", "l2_shuffled",
            "cos_delta", "cos_delta_shuffled", "action_sens"]
    for sname, st in results.items():
        T = st["steps"]
        show = list(range(T)) if T <= 6 else sorted({0, 1, 2, T // 2, T - 1})
        head = f"{name} @ {step}  {sname}  (n={st['n']}, dim={st['dim']}, {T} future steps)"
        print("\n" + head)
        print("-" * len(head))
        print(f"{'':20s}" + "".join(f"{'t=' + str(i + 1):>10s}" for i in show))
        for r in rows:
            if r in st:
                print(f"{r:20s}" + "".join(f"{st[r][i]:10.4f}" for i in show))
        print(f"{'pred/persist':20s}" + "".join(
            f"{st['l2_pred'][i] / max(st['l2_persist'][i], 1e-12):10.2f}" for i in show))
        print(f"pred_var_ratio {st['pred_var_ratio']:.3f}   ->  {st['verdict']}")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", required=True,
                   help="checkpoints: <entity>/<project>/<id>[:alias], <id>[:alias], or a .pt path")
    p.add_argument("--project", default=DEFAULT_PROJECT, help="project for bare run ids")
    p.add_argument("--n", type=int, default=64, help="held-out windows per checkpoint")
    p.add_argument("--bs", type=int, default=8, help="batch size (>= 2 for the shuffled baseline)")
    p.add_argument("--seed", type=int, default=0, help="which windows and which shuffles")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--device", default="auto", help="auto picks cuda, then cpu")
    p.add_argument("--workers", type=int, default=4, help="video-decode workers")
    p.add_argument("--dataset-root", default=None,
                   help="override the run's dataset.root (the training box's path)")
    p.add_argument("--from-hub", action="store_true",
                   help="drop dataset.root and let lerobot fetch the repo into the HF cache")
    p.add_argument("--ckpt-dir", default="outputs/served", help="wandb download dir")
    p.add_argument("--out", default="/tmp/probe_dynamics")
    args = p.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from lewam.datasets import build_dataset

    cache = {}
    for ref in args.run:
        t0 = time.time()
        model, cfg, step, name, path = load_model(ref, device, args.ckpt_dir, args.project)
        log(f"{name} step {step} from {path} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
        dcfg = cfg.dataset
        if args.from_hub:
            dcfg.root = None
        elif args.dataset_root:
            dcfg.root = args.dataset_root
        key = json.dumps(dict(dcfg), default=str, sort_keys=True) + args.split
        if key not in cache:
            ds = build_dataset(dcfg, norm_stats_override=model.norm_stats, augment=False,
                               split=args.split)
            idx, eps = pick_windows(ds, args.n, args.seed)
            items = decode_windows(ds, idx, args.workers)
            log(f"decoded {len(items)} {args.split} windows from {len(set(eps))} episodes "
                f"({len(ds)} in split)")
            cache[key] = (unwrap(ds), items)
        source, items = cache[key]
        results = summarize(probe(model, items, source, args.bs, device, args.seed), args.seed)
        print_table(name, step, results)
        rec = {"name": name, "ref": ref, "checkpoint": str(path), "step": step,
               "split": args.split, "seed": args.seed, "seconds": round(time.time() - t0, 1),
               "streams": results}
        (out / f"{name}_{step}.json").write_text(json.dumps(rec, indent=1))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
