"""Predicted vs recorded action chunks, drawn as end-effector paths in 3D.

`loss/int_action` is a mean over normalized joint angles: it says a chunk is off by 0.03
without saying whether the gripper went to the wrong place, arrived late, or took a
different route there. This runs a checkpoint on the HELD-OUT split and pushes both the
predicted chunk and the recorded one through the arm's forward kinematics, so the
comparison is in centimetres of tool position instead.

The FK is the manufacturer's URDF geometry that `gravity_model.link_frames` already
carries (the same one gravity compensation runs on). Joint angles are degrees in the
dataset and radians there, and only the 6 revolute joints are kinematic -- the 7th action
is the gripper, plotted as marker size rather than position.

    python -m lewam.debug.chunk_fk --run halicki/lewam-testing/u6vblv66:best
    python -m lewam.debug.chunk_fk --run <id> --show           # rotate the 3D panels
    python -m lewam.debug.chunk_fk --run <id> --num-flow-steps 20
    python -m lewam.debug.chunk_fk --run <id> --samples 12 --out /tmp/fk
    python -m lewam.debug.chunk_fk --run <id> --device cuda --seed 3

Writes <out>_paths.png (a 3D panel per sample) and <out>_error.png (tool error against
step within the chunk, over every sample), and prints the same error summary. `--show`
opens the figures instead, so the 3D panels can be rotated -- a fixed viewpoint hides
whether a predicted path is a coherent motion seen edge-on or a scribble. It needs a
display, so it is off by default: this repo's debug scripts also run on a headless box.
"""

import argparse

import matplotlib

import numpy as np
import torch

from lewam.utils.spec import parse_index, raw_index


def _fk_positions(chunk_deg):
    """(T, >=6) joint angles in DEGREES -> (T, 3) tool origin in metres.

    The last link frame is the gripper body: `LINKS[-1]` is the gripper assembly bolted
    to link6, so its origin is the tool point the arm actually moves through space.
    """
    from lerobot.robots.rebot_b601_follower.gravity_model import NUM_JOINTS, link_frames

    out = np.empty((len(chunk_deg), 3), dtype=float)
    for t, row in enumerate(chunk_deg):
        frames = link_frames(np.deg2rad(np.asarray(row, dtype=float)[:NUM_JOINTS]))
        out[t] = frames[-1][1]
    return out


def load(run, device, ckpt_dir="outputs/served"):
    from pathlib import Path

    from lewam.algorithms import build_algorithm
    from lewam.utils.checkpoint import load_checkpoint
    from lewam.utils.ckpt_utils import (config_from_checkpoint, download_latest_checkpoint,
                                        fetch_run_config, split_alias)

    run_path, alias = split_alias(run)
    path = download_latest_checkpoint(run_path, Path(ckpt_dir), alias=alias)
    # the embedded config preserves declaration order exactly as trained; wandb's copy
    # is the fallback for checkpoints from before configs were embedded
    stored = config_from_checkpoint(path)
    if stored is None:
        stored = fetch_run_config(run_path)
    model = build_algorithm(stored.algorithm)
    step = load_checkpoint(model, None, path)
    return model.to(device).eval(), stored, step


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="<entity>/<project>/<run_id>[:alias]")
    p.add_argument("--samples", type=int, default=8, help="held-out windows to draw")
    p.add_argument("--seed", type=int, default=0, help="which windows; same split either way")
    p.add_argument("--device", default="auto", help="auto picks cuda, then cpu")
    p.add_argument("--out", default="/tmp/chunk_fk")
    p.add_argument("--num-flow-steps", type=int, default=None,
                   help="override the run's ODE step count at inference (flow arms only; "
                        "the shipped arms solve with 4, which is few next to pi0's 10). "
                        "Ignored with a warning on a policy that does not integrate")
    p.add_argument("--show", action="store_true",
                   help="open the figures interactively (rotatable 3D) instead of only "
                        "writing PNGs; needs a display")
    p.add_argument("--dataset-root", default=None,
                   help="override the run's dataset.root, which is the training box's path")
    args = p.parse_args()

    # picked AFTER parsing: importing pyplot fixes the backend, and Agg is the only one
    # that works on a headless box, so it cannot be the unconditional default here
    if not args.show:
        matplotlib.use("Agg")
    global plt
    import matplotlib.pyplot as plt

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto"
        else args.device
    )
    model, stored, step = load(args.run, device)
    print(f"{args.run} @ step {step} on {device}")
    if args.num_flow_steps is not None:
        if hasattr(model, "num_flow_steps"):
            print(f"num_flow_steps {model.num_flow_steps} -> {args.num_flow_steps}")
            model.num_flow_steps = args.num_flow_steps
        else:
            print(f"--num-flow-steps ignored: {type(model).__name__} does not integrate "
                  f"an ODE (ACT and diffusion carry their own samplers)")

    from lewam.datasets import build_dataset
    from lewam.utils.normalize import Normalizer

    dcfg = stored.dataset
    if args.dataset_root:
        dcfg = dcfg.copy()
        dcfg.root = args.dataset_root
    # the stored root is the TRAINING box's path; on any other machine lerobot would try
    # to mkdir it (and on this one, into the home dir). Same fallback lerobot_policy uses:
    # drop to the hub cache, which downloads on first use
    from pathlib import Path as _Path
    root = dcfg.get("root")
    if root is not None and not _Path(root).exists():
        print(f"dataset.root '{root}' not present here; falling back to the hub cache "
              f"(pass --dataset-root for a local copy)")
        dcfg = dcfg.copy()
        dcfg.root = None
    stats = getattr(model, "norm_stats", None)
    # the stats the weights were trained with, never stats recomputed from held-out
    # episodes -- the same rule offline eval follows
    dataset = build_dataset(dcfg, norm_stats_override=stats, augment=False, split="val")
    if not len(dataset):
        raise SystemExit("held-out split is empty; check dataset.validation_split")
    normalizer = Normalizer(stats, method=getattr(model, "norm_method", "mean_std")) if stats else None
    print(f"held-out windows: {len(dataset)}")

    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(dataset), size=min(args.samples, len(dataset)), replace=False)
    field = model.action_field

    truths, preds = [], []
    for idx in picks:
        item = dataset[int(idx)]
        obs = {k: v.unsqueeze(0).to(device) for k, v in item.items() if torch.is_tensor(v)}
        with torch.no_grad():
            out = model.predict(obs)
        pred = out[model.action_stream] if isinstance(out, dict) else out
        # Align the target on the predicted stream's OWN row layout. The field carries the
        # union of every entry reading it, so a flow arm's 36 predicted rows are the tail of
        # a 72-row window (36 of a2a history first), while lerobot's diffusion declares
        # "-1..38" -- 40 rows, because its horizon is measured from the first observation --
        # and returns only n_action_steps of them, starting at delta 0. Taking the last
        # len(pred) rows is right for the first and silently off by one for the second.
        rows = parse_index(raw_index(model.predict_spec[model.action_stream]))
        window = item[field][-len(rows):]
        start = rows.index(0) if 0 in rows else 0
        truth = window[start:start + pred.shape[1]].unsqueeze(0).to(device)
        if truth.shape[1] != pred.shape[1]:
            raise SystemExit(
                f"predicted {pred.shape[1]} steps but only {truth.shape[1]} target rows "
                f"exist from index 0 in a window of {rows}; the run's predict spec and its "
                f"policy disagree about the chunk"
            )
        if normalizer is not None:
            pred = normalizer.to(device).unnormalize(field, pred)
            truth = normalizer.unnormalize(field, truth)
        preds.append(pred[0].cpu().numpy())
        truths.append(truth[0].cpu().numpy())

    fk_t = [_fk_positions(c) for c in truths]
    fk_p = [_fk_positions(c) for c in preds]
    err = np.stack([np.linalg.norm(a - b, axis=-1) for a, b in zip(fk_t, fk_p)])  # (S, T) m

    # err.shape[1], not chunk_len: a policy may return fewer steps than its predict window
    # declares (lerobot diffusion returns n_action_steps out of a 40-row horizon)
    print(f"\ntool-position error over {len(picks)} held-out chunks, {err.shape[1]} steps:")
    print(f"  mean   {err.mean() * 100:7.2f} cm")
    print(f"  median {np.median(err) * 100:7.2f} cm")
    print(f"  final  {err[:, -1].mean() * 100:7.2f} cm   (end of the chunk)")
    print(f"  max    {err.max() * 100:7.2f} cm")

    cols = min(4, len(picks))
    rows = -(-len(picks) // cols)
    fig = plt.figure(figsize=(4.0 * cols, 3.8 * rows))
    for i, (t, q) in enumerate(zip(fk_t, fk_p)):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.plot(*t.T, color="tab:blue", lw=2.0, label="recorded")
        ax.plot(*q.T, color="tab:red", lw=2.0, ls="--", label="predicted")
        ax.scatter(*t[0], color="k", s=28, label="chunk start")
        ax.scatter(*t[-1], color="tab:blue", s=28, marker="*")
        ax.scatter(*q[-1], color="tab:red", s=28, marker="*")
        ax.set_title(f"window {picks[i]}   final {err[i, -1] * 100:.1f} cm", fontsize=9)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.set_zlabel("z (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        # one scale for every panel, so a small error cannot look large by autoscaling
        span = np.stack([t, q])
        centre = span.reshape(-1, 3).mean(0)
        half = max(float(np.abs(span.reshape(-1, 3) - centre).max()), 0.02) * 1.15
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            setter(c - half, c + half)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{args.run} @ step {step} — tool path, recorded vs predicted "
                 f"({err.shape[1]}-step chunk)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{args.out}_paths.png", dpi=140)
    print(f"\nwrote {args.out}_paths.png")
    if not args.show:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    steps = np.arange(err.shape[1])
    for row in err:
        ax.plot(steps, row * 100, color="tab:red", alpha=0.25, lw=1.0)
    ax.plot(steps, err.mean(0) * 100, color="k", lw=2.2, label="mean")
    ax.plot(steps, np.median(err, axis=0) * 100, color="tab:blue", lw=1.6, ls="--",
            label="median")
    ax.set_xlabel("step within the predicted chunk")
    ax.set_ylabel("tool position error (cm)")
    ax.set_title(f"{args.run} @ step {step} — error grows with horizon", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}_error.png", dpi=140)
    print(f"wrote {args.out}_error.png")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
