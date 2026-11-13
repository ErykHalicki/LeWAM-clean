"""One-off: pull a ~1GB random subset of episodes from the hosted tworoom.h5 (12.7GB)
into $STABLEWM_HOME/datasets/tworoom_1gb.h5, for fast local iteration on the LeWAM arch
without downloading the full dataset. Episodes are sampled randomly across the whole
file (not a prefix) -- a prefix would badly undersample per-episode-varying fields like
the target/door layout, the same mistake normalize.py's compute_stats used to make.
"""

import time

import hdf5plugin  # noqa: F401 -- registers compression filters used by the source file
import fsspec
import h5py
import numpy as np
from huggingface_hub import hf_hub_url
from stable_worldmodel.data.formats.hdf5 import HDF5Writer
from stable_worldmodel.data.utils import get_cache_dir

REPO = "MinghaoFu/lewm-official-data"
SRC_NAME = "tworoom"
OUT_NAME = "tworoom_1gb"
TARGET_BYTES = 1 * 1024**3
COLUMNS = ["pixels", "action", "proprio", "observation"]
SEED = 0


def main():
    url = hf_hub_url(repo_id=REPO, repo_type="dataset", filename=f"{SRC_NAME}.h5")
    # blockcache: reads in 8MB blocks, so h5py's many small chunked reads mostly hit
    # the local cache instead of one HTTP request each
    of = fsspec.open(url, mode="rb", cache_type="blockcache", block_size=8 * 1024 * 1024)
    src = h5py.File(of.open(), "r")

    ep_len = src["ep_len"][:]
    ep_offset = src["ep_offset"][:]
    n_ep = len(ep_len)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_ep)

    bytes_per_row = sum(
        int(np.prod(src[c].shape[1:])) * src[c].dtype.itemsize for c in COLUMNS
    )
    print(f"columns: {COLUMNS}, ~{bytes_per_row} bytes/row uncompressed", flush=True)

    out_path = get_cache_dir(None, sub_folder="datasets") / f"{OUT_NAME}.h5"

    # resumable: a prior run's episodes are exactly order[:already_done] (same seed
    # -> same permutation). This file's per-row-chunked layout stalls unpredictably
    # over HTTP, so a single unbroken run is not reliable.
    already_done, already_bytes = 0, 0
    if out_path.exists():
        with h5py.File(out_path, "r") as existing:
            already_done = existing["ep_len"].shape[0]
            already_bytes = int(existing["ep_len"][:].sum()) * bytes_per_row
        print(f"resuming: {already_done} episodes already written, ~{already_bytes / 1024**2:.0f} MiB", flush=True)

    written_bytes = already_bytes
    n_written = already_done
    mode = "append" if out_path.exists() else "overwrite"
    with HDF5Writer(out_path, mode=mode) as w:
        for ei in order[already_done:]:
            if written_bytes >= TARGET_BYTES:
                break
            t0 = time.time()
            off, ln = int(ep_offset[ei]), int(ep_len[ei])
            ep_data = {c: src[c][off : off + ln] for c in COLUMNS}
            w.write_episode(ep_data)
            written_bytes += ln * bytes_per_row
            n_written += 1
            print(
                f"  ep {n_written} (len={ln}) in {time.time() - t0:.1f}s, "
                f"~{written_bytes / 1024**2:.0f} MiB",
                flush=True,
            )

    if written_bytes >= TARGET_BYTES:
        print(f"done: {n_written} episodes, ~{written_bytes / 1024**3:.2f} GiB -> {out_path}", flush=True)
    else:
        print(f"stopped early: {n_written} episodes, ~{written_bytes / 1024**2:.0f} MiB -> {out_path} (rerun to continue)", flush=True)


if __name__ == "__main__":
    main()
