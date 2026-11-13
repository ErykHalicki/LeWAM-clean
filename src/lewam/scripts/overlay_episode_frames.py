#!/usr/bin/env python3
"""Overlay the first frame of every episode into one image, and the last frames into
another, so a dataset's start and end distributions can be read at a glance.

What the pictures say: anything sharp is the same in every episode (the table, the arm's
home pose, a fixture that never moves), and anything blurred varies. A start overlay with
a crisp object means the scene was reset to the same spot every time -- a policy trained
on it has seen one initial condition, not a distribution. A smeared end overlay means the
task finished in a different place each time, which is either genuine task variety or an
inconsistent success criterion.

One image per camera per endpoint, written to --out:

    <camera>_first.png   every episode's first frame, combined
    <camera>_last.png    every episode's last frame, combined

    python src/lewam/scripts/overlay_episode_frames.py ehalicki/b601_towel_fold
    python src/lewam/scripts/overlay_episode_frames.py <repo_id> --cameras zed_left --mode median

Four ways to combine them, all reading the same frames:

    mean    (default) the literal overlay -- every episode contributes equally, so a rare
            start still tints the image. Shows how wide the spread is, at the cost of
            smearing whatever moved across everywhere it went.
    median  the typical frame, outliers dropped. A clean backdrop to compare mean against.
    stack   each episode's moving parts at full opacity over the median background, so
            four block positions read as four solid blocks instead of four faint ghosts.
    blend   the same foreground, averaged where episodes overlap rather than last-wins.
    spread  a heat map of which pixels vary at all, over a dimmed background.

Reach for `stack` when the mean is fog and you want to see the actual states; `spread`
when the question is which parts of the scene are the variable ones.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo_id", help="lerobot dataset, e.g. ehalicki/b601_towel_fold")
    p.add_argument("--root", default=None, help="local dataset root (default: the HF cache)")
    p.add_argument("--revision", default=None, help="dataset revision")
    p.add_argument("--cameras", default=None,
                   help="comma-separated camera keys (default: every video/image feature)")
    p.add_argument("--episodes", type=int, default=None,
                   help="use only the first N episodes (default: all)")
    p.add_argument("--mode", default="mean", choices=("mean", "median", "stack", "spread", "blend"),
                   help="how to combine the frames (default: %(default)s)")
    p.add_argument("--threshold", type=int, default=30,
                   help="stack/blend modes: channel difference from the background that counts "
                        "as foreground, 0-255 (default: %(default)s)")
    p.add_argument("--out", default=None,
                   help="output directory (default: overlays/<dataset name>)")
    # pyav rather than lerobot's default: it picks torchcodec whenever the package
    # imports, and torchcodec's wheel links against CUDA libraries a robot box lacks
    p.add_argument("--video-backend", default="pyav",
                   help="lerobot video decoder (default: %(default)s)")
    # matches the dataset configs. Re-encoding re-quantizes frame pts by ~1e-4 s,
    # right at lerobot's default tolerance, so the default rejects frames that are fine
    p.add_argument("--tolerance-s", type=float, default=0.005,
                   help="lerobot timestamp tolerance (default: %(default)s)")
    return p.parse_args()


def episode_bounds(meta, limit):
    """(first, last) global frame index per episode. LeRobotDataset indexes frames
    continuously across episodes, so an episode's own frames start at the running sum
    of every earlier episode's length."""
    lengths = np.asarray(meta.episodes["length"])
    if limit is not None:
        lengths = lengths[:limit]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    return [(int(s), int(s + n - 1)) for s, n in zip(starts, lengths, strict=True) if n > 0]


def to_hwc_uint8(tensor):
    """A lerobot visual frame (C, H, W) -> (H, W, C) uint8, whatever dtype it arrived in."""
    array = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0 if array.max() <= 1.0 else array, 0, 255).astype(np.uint8)
    return array


def _deviation(frames, background):
    """Per-frame, per-pixel distance from the background, as the largest difference
    across the colour channels. Computed a frame at a time: the whole stack in float
    is ~145MB for 48 frames of a 376x672 camera, and nothing here needs it at once."""
    base = background.astype(np.int16)
    return np.stack([
        np.abs(frame.astype(np.int16) - base).max(axis=-1) for frame in frames
    ])


def combine(frames, mode, threshold):
    """Stacked frames -> one image.

    mean/median average the episodes together, so anything that moves is spread over
    every place it went. The other two keep the static scene as a backdrop and treat
    the moving pixels as the subject instead:

    stack   every episode's foreground painted at FULL opacity over the median
            background, so four block positions read as four solid blocks rather than
            four 25%-opacity ghosts. Where two episodes overlap the later one wins, so
            this shows the set of positions, not how many episodes are in each.
    blend   like stack, but overlapping foregrounds are averaged instead of the last one
            winning -- no episode is privileged, at the cost of softer edges wherever
            the states disagree.
    spread  how much each pixel varies at all, as heat over a dimmed greyscale
            background. Answers "which parts of the scene are the variable ones"
            without claiming anything about what the values were.
    """
    stacked = np.stack(frames)
    if mode == "mean":
        return stacked.astype(np.float32).mean(axis=0).round().clip(0, 255).astype(np.uint8)

    background = np.median(stacked, axis=0).round().astype(np.uint8)
    if mode == "median":
        return background

    if mode in ("stack", "blend"):
        foreground = _deviation(frames, background) > threshold
        if mode == "stack":
            out = background.copy()
            for frame, differs in zip(frames, foreground, strict=True):
                out[differs] = frame[differs]
            return out
        # mean over the episodes with foreground at each pixel, not over all of them:
        # that drops the frequency weighting, so a state one episode visited is as
        # solid as one twenty visited
        total = np.zeros(background.shape, np.float32)
        count = np.zeros(background.shape[:2], np.int32)
        for frame, differs in zip(frames, foreground, strict=True):
            total[differs] += frame[differs]
            count[differs] += 1
        averaged = total / np.maximum(count, 1)[..., None]
        return np.where(count[..., None] > 0, averaged, background).round().astype(np.uint8)

    # scaled against its 99th percentile so one hot pixel cannot flatten everything
    # else, and gamma-lifted so moderate variation still shows
    variation = stacked.astype(np.float32).std(axis=0).max(axis=-1)
    ceiling = np.percentile(variation, 99) or 1.0
    heat = np.clip(variation / ceiling, 0, 1)[..., None] ** 0.5
    grey = background.astype(np.float32).mean(axis=-1, keepdims=True) * 0.5
    hot = np.array([255.0, 80.0, 0.0]) * (1 - heat) + np.array([255.0, 255.0, 180.0]) * heat
    return (grey * (1 - heat) + hot * heat).round().clip(0, 255).astype(np.uint8)


def main():
    args = parse_args()
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from PIL import Image

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root, revision=args.revision)
    visual = [k for k, v in meta.features.items() if v["dtype"] in ("video", "image")]
    cameras = args.cameras.split(",") if args.cameras else visual
    unknown = [c for c in cameras if c not in visual]
    if unknown:
        sys.exit(f"no such camera(s) {unknown}. this dataset has: {visual}")
    if not cameras:
        sys.exit(f"{args.repo_id} has no video or image features to overlay")

    bounds = episode_bounds(meta, args.episodes)
    if not bounds:
        sys.exit(f"{args.repo_id} has no episodes")

    dataset = LeRobotDataset(
        args.repo_id, root=args.root, revision=args.revision, return_uint8=True,
        video_backend=args.video_backend, tolerance_s=args.tolerance_s,
    )

    # held as uint8 until the combine: ~76MB per endpoint for 100 episodes of a
    # 376x672 camera, against the ~300MB a float accumulator would cost
    frames = {(cam, end): [] for cam in cameras for end in ("first", "last")}
    for n, (first, last) in enumerate(bounds, 1):
        for end, idx in (("first", first), ("last", last)):
            item = dataset[idx]
            for cam in cameras:
                frames[(cam, end)].append(to_hwc_uint8(item[cam]))
        print(f"\r  read {n}/{len(bounds)} episodes", end="", flush=True)
    print()

    out = Path(args.out) if args.out else Path("overlays") / args.repo_id.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    for (cam, end), collected in frames.items():
        path = out / f"{cam.split('.')[-1]}_{end}.png"
        Image.fromarray(combine(collected, args.mode, args.threshold)).save(path)
        print(f"  {path}  ({len(collected)} episodes, {args.mode})")


if __name__ == "__main__":
    main()
