"""Reclaim disk from downloaded model checkpoints, interactively.

Three places accumulate model weights on a machine that has evaluated or served runs:

    outputs/served/        post-hoc downloads (chunk_fk, serve.py) -- one per run served,
                           ~800 MB each for the b601 arms
    outputs/checkpoints/   training / resume checkpoints, one dir per wandb run id
    <wandb cache>/artifacts  wandb's content-addressed artifact cache: every version of
                           every artifact ever downloaded stays here until deleted, and
                           it is usually the real space hog

Everything listed is re-downloadable from wandb (or rebuilt by training), so deletion
costs nothing but a future re-download. The script lists what it finds, shows the total,
and asks before touching anything:

    python -m lewam.scripts.clean_checkpoints
    python -m lewam.scripts.clean_checkpoints --yes             # no prompt (cron/CI)
    python -m lewam.scripts.clean_checkpoints --keep-wandb-cache
    python -m lewam.scripts.clean_checkpoints --root ~/other/LeWAM
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def dir_size(path):
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            fp = Path(dirpath) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024


def wandb_cache_dir():
    """Where wandb keeps its artifact cache on this machine: WANDB_CACHE_DIR if set
    (main.py points it at a tmp scratch dir), else the platform default."""
    env = os.environ.get("WANDB_CACHE_DIR")
    if env:
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "wandb"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "wandb"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".",
                   help="repo root whose outputs/ to scan (default: cwd)")
    p.add_argument("--yes", action="store_true", help="delete without prompting")
    p.add_argument("--keep-wandb-cache", action="store_true",
                   help="leave the wandb artifact cache alone")
    args = p.parse_args()
    root = Path(args.root).expanduser().resolve()

    targets = []  # (label, path, n_checkpoints, bytes)
    for sub in ("outputs/served", "outputs/checkpoints"):
        d = root / sub
        if d.is_dir():
            pts = list(d.rglob("*.pt"))
            if pts:
                targets.append((sub, d, len(pts), sum(f.stat().st_size for f in pts)))
    if not args.keep_wandb_cache:
        cache = wandb_cache_dir() / "artifacts"
        if cache.is_dir():
            size = dir_size(cache)
            if size:
                # content-addressed blobs, not .pt files; count the artifact objects
                n = sum(1 for f in cache.rglob("*") if f.is_file())
                targets.append(("wandb artifact cache", cache, n, size))

    if not targets:
        print("no downloaded checkpoints found -- nothing to clean")
        return

    total = 0
    for label, path, n, size in targets:
        what = "files" if label == "wandb artifact cache" else "checkpoints"
        print(f"{n} {what} found in {path}  ({human(size)})")
        total += size
    print(f"Deletion will free up {human(total)}")
    print("(everything here re-downloads from wandb on next use)")

    if not args.yes:
        try:
            answer = input("Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\naborted")
            return
        if answer not in ("y", "yes"):
            print("aborted")
            return

    for label, path, _n, size in targets:
        shutil.rmtree(path, ignore_errors=True)
        print(f"deleted {path}  ({human(size)})")
    print(f"freed {human(total)}")


if __name__ == "__main__":
    main()
