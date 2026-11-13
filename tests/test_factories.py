import pytest
from omegaconf import OmegaConf

from lewam.algorithms import build_algorithm
from lewam.datasets import build_dataset
from lewam.experiments import build_experiment
from lewam.experiments.eval import build_eval


def test_unknown_dataset_backend_hints_extra():
    cfg = OmegaConf.create({"backend": "nope"})
    with pytest.raises(ValueError, match=r"lewam|nope|install"):
        build_dataset(cfg)


def test_unknown_eval_backend_hints_extra():
    cfg = OmegaConf.create({"backend": "nope"})
    with pytest.raises(ValueError, match=r"nope|install"):
        build_eval(cfg)


def test_unknown_algorithm_raises():
    cfg = OmegaConf.create({"name": "nope"})
    with pytest.raises(ValueError, match="nope"):
        build_algorithm(cfg)


def test_build_experiment_returns_registered_class():
    cfg = OmegaConf.create({"experiment": {"_name": "smoke"}})
    exp = build_experiment(cfg, output_dir="runs")
    assert exp.__class__.__name__ == "SmokeExperiment"


def test_unknown_experiment_raises():
    cfg = OmegaConf.create({"experiment": {"_name": "nope"}})
    with pytest.raises(ValueError, match="nope"):
        build_experiment(cfg)


def test_every_experiment_yaml_is_registered():
    # a yaml in configs/experiment/ missing from exp_registry fails only at runtime
    # on the cluster, so pin the mapping here
    from pathlib import Path

    from lewam.experiments import exp_registry

    yamls = {p.stem for p in (Path(__file__).parent.parent / "configs" / "experiment").glob("*.yaml")}
    missing = yamls - set(exp_registry)
    assert not missing, f"experiment yamls without a registry entry: {sorted(missing)}"
