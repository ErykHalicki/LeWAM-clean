import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("stable_worldmodel")

from lewam.datasets import build_dataset  # noqa: E402


def _write_fake_episode(writer, length=6):
    writer.write_episode(
        {
            "pixels": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(length)],
            "action": [np.zeros(2, dtype=np.float32) for _ in range(length)],
            "proprio": [np.zeros(4, dtype=np.float32) for _ in range(length)],
            "state": [np.zeros(7, dtype=np.float32) for _ in range(length)],
        }
    )


def test_swm_source_emits_standard_batch(tmp_path):
    # fake HDF5 matching the real pusht_expert_train.h5 schema, so this exercises
    # SWMSource without the 43GB download
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    h5_path = tmp_path / "datasets" / "fake_pusht.h5"
    with HDF5Writer(h5_path) as w:
        _write_fake_episode(w)
        _write_fake_episode(w)

    cfg = OmegaConf.create(
        {
            "backend": "swm",
            "hf_repo": "unused/should-not-be-fetched",
            "name": "fake_pusht",
            "cache_dir": str(tmp_path),
            "conditioning": {
                "observation.state": {"index": [0]},
                "observation.images.pixels": {"index": [0]},
            },
            "predict": {"action": {"index": [0, 1]}},
        }
    )

    dataset = build_dataset(cfg)
    # the terminal row of each episode holds the NaN "no action" sentinel: length 6,
    # action max_offset 1+1 -> 4 decision points per episode
    assert len(dataset) == 8

    batch = dataset[0]
    assert set(batch) <= dataset.provided_modalities
    assert {"observation.env_state", "state", "proprio", "pixels"} <= dataset.provided_modalities
    assert batch["observation.images.pixels"].shape == (1, 3, 8, 8)
    assert batch["observation.state"].shape == (1, 4)
    assert batch["action"].shape == (2, 2)


def test_swm_source_windows_by_from_field_union(tmp_path):
    # three entries read observation.state with different windows, so its batch field
    # must carry the sorted union -2..1; unreferenced columns are not loaded
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    h5_path = tmp_path / "datasets" / "fake_pusht.h5"
    with HDF5Writer(h5_path) as w:
        _write_fake_episode(w)
        _write_fake_episode(w)

    cfg = OmegaConf.create(
        {
            "backend": "swm",
            "name": "fake_pusht",
            "cache_dir": str(tmp_path),
            "conditioning": {
                "state_context": {"from": "observation.state", "index": "-2..-1"},
                "state": {"via": "cross", "from": "observation.state", "index": [0]},
            },
            "predict": {
                "future_state": {"from": "observation.state", "index": "0..1"},
                "action": {"from": "action", "index": "0..1"},
            },
        }
    )

    dataset = build_dataset(cfg)
    assert len(dataset) == 4

    batch = dataset[0]
    assert set(batch) == {"observation.state", "action"}
    assert batch["observation.state"].shape == (4, 4)
    assert batch["action"].shape == (2, 2)


def test_swm_source_appends_episode_final_frame_for_last(tmp_path):
    # an `index: "last"` entry makes the field carry the episode-final frame appended
    # after the union of relative windows
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer

    h5_path = tmp_path / "datasets" / "fake_tworoom.h5"
    with HDF5Writer(h5_path) as w:
        for _ in range(2):
            w.write_episode(
                {
                    "pixels": [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(6)],
                    "action": [np.zeros(2, dtype=np.float32) for _ in range(6)],
                }
            )

    cfg = OmegaConf.create(
        {
            "backend": "swm",
            "name": "fake_tworoom",
            "cache_dir": str(tmp_path),
            "conditioning": {
                "pixel_context": {"from": "observation.images.pixels", "index": "-1..0"},
                "goal": {"from": "observation.images.pixels", "index": "last", "via": "cross"},
            },
            "predict": {
                "future_video": {"from": "observation.images.pixels", "index": "1"},
                "action": {"from": "action", "index": "0..1"},
            },
        }
    )

    dataset = build_dataset(cfg)
    batch = dataset[0]
    pixels = batch["observation.images.pixels"]
    assert pixels.shape == (4, 3, 8, 8)
    assert (pixels[0] == 0).all()
    assert (pixels[-1] == 5).all()
