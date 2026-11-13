import os

import numpy as np
import torch

try:
    # registers the pixel column's compression filter with h5py, in every process
    # that unpickles this dataset (DataLoader workers included)
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

from ..utils.spec import LAST, flow_source_entries, raw_index, spec_fields, split_index
from .base import BaseSource

_COLUMNS = {
    "observation.images.pixels": "pixels",
    "observation.state": "proprio",
    "observation.env_state": "state",
    "action": "action",
}


class SWMSource(BaseSource):
    """Expert demonstrations via stable-worldmodel's HDF5 format, windowed by the
    temporal index spec (`cfg.conditioning` / `cfg.predict`, see utils/spec.py). Each
    spec entry reads the batch field named by its `from` key (default: the entry name),
    a window of relative step offsets around a sampled decision point t=0. A field read
    by several entries carries the sorted union of their windows, and each entry slices
    its own window back out (see algorithms/predictive_model.py).

    Reads `$STABLEWM_HOME/datasets/<name>.h5`, downloading it from the configured HF
    dataset repo on first use if not already cached.

    `frameskip` (default 1) sets the temporal abstraction le-wm trains with (theirs: 5):
    one model step spans `frameskip` raw env frames, so a spec offset of s reads the raw
    frame at t + s*frameskip, and the action field of a model step is the CONCATENATION
    of that step's `frameskip` raw actions (dim frameskip * action_dim) -- the spec's
    action `dim` must match. Decision points t remain dense over raw frames.
    """

    provided_modalities = set(_COLUMNS)

    def __init__(self, cfg):
        import h5py
        from stable_worldmodel.data.utils import get_cache_dir

        name = cfg.name
        size = cfg.get("image_size")
        self._image_size = tuple(int(v) for v in size) if size else None
        self._frameskip = int(cfg.get("frameskip", 1))
        cache_dir = cfg.get("cache_dir")
        datasets_dir = get_cache_dir(cache_dir, sub_folder="datasets")
        self._h5_path = datasets_dir / f"{name}.h5"
        if not self._h5_path.exists():
            hf_repo = cfg.get("hf_repo")
            if not hf_repo:
                raise FileNotFoundError(
                    f"{self._h5_path} not found locally and no hf_repo configured to fetch it."
                )
            self._download(hf_repo, name, datasets_dir)

        self._h5 = None
        self._h5_pid = None
        with h5py.File(self._h5_path, "r", swmr=True) as f:
            lengths, ep_offsets = f["ep_len"][:], f["ep_offset"][:]
            total = int(lengths.sum())
            self._step_columns = {
                k for k in f.keys() if f[k].shape and f[k].shape[0] == total
            }
        self.provided_modalities = {
            alias for alias, col in _COLUMNS.items() if col in self._step_columns
        } | self._step_columns

        windows, has_last = {}, {}
        entries = {**cfg.conditioning, **cfg.predict, **flow_source_entries(cfg.predict)}
        for key, spec in entries.items():
            for field in spec_fields(key, spec):
                if _COLUMNS.get(field, field) in self._step_columns and "index" in spec:
                    rel, hl = split_index(raw_index(spec))
                    windows.setdefault(field, set()).update(rel)
                    has_last[field] = has_last.get(field, False) or hl
        # "last" is appended after the relative offsets, matching predictive_model.py
        self._windows = {
            field: sorted(steps) + ([LAST] if has_last.get(field) else [])
            for field, steps in windows.items()
        }
        # validity comes from the relative offsets only; "last" always exists. Offsets
        # are in MODEL steps of `frameskip` raw frames each.
        fs = self._frameskip
        if fs > 1:
            for field, steps in self._windows.items():
                assert not (_COLUMNS.get(field, field) == "action" and LAST in steps), (
                    "frameskip > 1 cannot serve the episode-final ('last') frame for an "
                    "action field: the final row is the NaN sentinel, not a full block"
                )
        min_offset = min(s for steps in windows.values() for s in steps) * fs
        # every episode's terminal row has a NaN `action` sentinel, so no window may
        # land on it; an action BLOCK spans fs rows, all of which must precede it
        max_offset = max(
            (max(steps) * fs + fs if _COLUMNS.get(field, field) == "action" else max(steps) * fs)
            for field, steps in windows.items() if steps
        )

        self._ep_offsets = ep_offsets
        self._ep_lengths = lengths
        self._index = [
            (ep, t)
            for ep, length in enumerate(lengths)
            for t in range(-min_offset, length - max_offset)
        ]

    @staticmethod
    def _download(hf_repo, name, datasets_dir):
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=hf_repo,
            repo_type="dataset",
            filename=f"{name}.h5",
            local_dir=datasets_dir,
        )

    def __getstate__(self):
        return {**self.__dict__, "_h5": None, "_h5_pid": None}

    def _file(self):
        import h5py

        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            self._h5 = h5py.File(self._h5_path, "r", swmr=True)
            self._h5_pid = pid
        return self._h5

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep, t = self._index[idx]
        f = self._file()
        offset = self._ep_offsets[ep]

        last_row = offset + int(self._ep_lengths[ep]) - 1
        fs = self._frameskip
        batch = {}
        for key, steps in self._windows.items():
            col = _COLUMNS.get(key, key)
            if col == "action" and fs > 1:
                data = np.stack([
                    np.concatenate(f[col][offset + t + s * fs: offset + t + s * fs + fs])
                    for s in steps
                ])
            else:
                rows = [last_row if s is LAST else offset + t + s * fs for s in steps]
                data = np.stack([f[col][r] for r in rows])
            tensor = torch.from_numpy(data)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2)
                if self._image_size and tuple(tensor.shape[-2:]) != self._image_size:
                    tensor = (
                        torch.nn.functional.interpolate(
                            tensor.float(), size=self._image_size, mode="area"
                        )
                        .round()
                        .clamp(0, 255)
                        .to(tensor.dtype)
                    )
            batch[key] = tensor
        return batch
