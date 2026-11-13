"""What a dataset config's `augment:` block actually does to its frames.

Pulls ONE episode video out of the dataset's repo and applies the configured
augmentations to a window of it, so a config can be read back visually before a run pays
to download the whole thing.

    python src/lewam/debug/show_augmentations.py b601_pnpt
    python src/lewam/debug/show_augmentations.py b601_towel_fold --draws 6
    python src/lewam/debug/show_augmentations.py b601_charger_pick_and_place \
        --field observation.images.wrist_cam --start 300 --out /tmp/wrist.png

Rows are independent draws, columns are the frames of one window, `--stride` raw frames
apart. A row is one training sample, whose draw is shared by the whole window -- so a crop
that slides along a row is a bug. A column is the spread the config admits.

Only the `streams:` patterns matching the chosen video field are applied; a config's
state-noise patterns have nothing to act on here and are listed as skipped.
"""

import argparse
from pathlib import Path

import torch


def dataset_cfg(name):
    """Compose one `configs/dataset/*.yaml` the way `main.py` would."""
    from hydra import compose, initialize_config_dir

    configs = Path(__file__).resolve().parents[3] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        # `conditioning`/`predict` interpolate into the algorithm, but OmegaConf resolves
        # lazily and nothing here reads them, so no algorithm need be selected
        return compose(config_name="base", overrides=[f"dataset={name}"]).dataset


def find_video(cfg, field=None):
    """(local path, field name) for one episode video, downloading it if it is not local."""
    from huggingface_hub import hf_hub_download, list_repo_files

    repo_id, root, revision = cfg.repo_id, cfg.get("root"), cfg.get("revision")
    if root and (local := sorted(Path(root).rglob("*.mp4"))):
        paths = [str(p.relative_to(root)) for p in local]
    else:
        paths = sorted(f for f in list_repo_files(repo_id, repo_type="dataset", revision=revision)
                       if f.endswith(".mp4"))
    if not paths:
        raise SystemExit(f"no .mp4 in '{repo_id}' -- is it an image-only dataset?")

    if field:
        paths = [p for p in paths if field in p.split("/")]
        if not paths:
            raise SystemExit(f"no video for field '{field}' in '{repo_id}'")
    chosen = paths[0]

    # the video key is a path component in every lerobot layout, and is the batch field
    # name the augment patterns match against
    parts = chosen.split("/")
    name = next((p for p in parts if p.startswith("observation.")), parts[-2])

    if root and (Path(root) / chosen).exists():
        return Path(root) / chosen, name
    return Path(hf_hub_download(repo_id, chosen, repo_type="dataset", revision=revision)), name


def read_window(path, start, count, stride):
    """(<=count, 3, H, W) uint8 frames from `start`, one every `stride`, and their offsets."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(path), dimension_order="NCHW")
    total = decoder.metadata.num_frames
    indices = list(range(start, min(start + count * stride, total), stride))
    if not indices:
        raise SystemExit(f"--start {start} is past the end of the video ({total} frames)")
    return decoder.get_frames_at(indices).data, [i - start for i in indices]


def resize(frames, image_size):
    """The dataset's own resize, so crop scales mean what they will at training time."""
    if not image_size or tuple(frames.shape[-2:]) == tuple(image_size):
        return frames
    out = torch.nn.functional.interpolate(frames.float(), size=tuple(image_size), mode="area")
    return out.round().clamp(0, 255).to(torch.uint8)


def build_augmenter(aug_cfg, field, seed):
    """An Augmenter carrying only the patterns that match `field`, plus the skipped ones.

    Augmenter raises on a pattern matching no field, which a one-field batch would trip on
    any config that also augments state.
    """
    import fnmatch

    from lewam.utils.augment import Augmenter

    streams = dict((aug_cfg or {}).get("streams") or {})
    matched = {p: ops for p, ops in streams.items() if fnmatch.fnmatchcase(field, p)}
    if not matched:
        raise SystemExit(
            f"no augment pattern matches '{field}'. patterns: {sorted(streams)}"
        )
    return Augmenter(matched, seed=seed), sorted(set(streams) - set(matched))


def plot(original, draws, offsets, field, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = 1 + len(draws), original.shape[0]
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.3 * rows), squeeze=False)
    for r, (label, window) in enumerate([("original", original), *[
            (f"draw {i}", d) for i, d in enumerate(draws)]]):
        for c in range(cols):
            ax = axes[r][c]
            ax.imshow(window[c].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if r == 0:
                ax.set_title(f"t+{offsets[c]}", fontsize=9)
    fig.suptitle(field, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset", help="a config name under configs/dataset/, without .yaml")
    p.add_argument("--field", help="which video stream (default: the first in the repo)")
    p.add_argument("--frames", type=int, default=4, help="window length (default 4)")
    p.add_argument("--stride", type=int, default=6, help="raw frames per step (default 6)")
    p.add_argument("--draws", type=int, default=4, help="independent draws (default 4)")
    p.add_argument("--start", type=int, default=0, help="first frame index (default 0)")
    p.add_argument("--seed", type=int, default=0, help="draw seed (default 0)")
    p.add_argument("--out", help="output png (default /tmp/augment_<dataset>.png)")
    args = p.parse_args()

    cfg = dataset_cfg(args.dataset)
    if not cfg.get("augment"):
        raise SystemExit(f"'{args.dataset}' sets `augment: null` -- nothing to show")

    path, field = find_video(cfg, args.field)
    print(f"repo    {cfg.repo_id}\nvideo   {path}\nfield   {field}")

    raw, offsets = read_window(path, args.start, args.frames, args.stride)
    frames = resize(raw, cfg.get("image_size"))
    augmenter, skipped = build_augmenter(cfg.augment, field, args.seed)
    ops = augmenter._ops_for(field)
    print(f"size    {tuple(frames.shape[-2:])}\nops     {ops}")
    if skipped:
        print(f"skipped {skipped} (no such field in a one-video batch)")

    draws = [augmenter({field: frames})[field] for _ in range(args.draws)]
    out = args.out or f"/tmp/augment_{args.dataset}.png"
    print(f"wrote   {plot(frames, draws, offsets, field, out)}")


if __name__ == "__main__":
    main()
