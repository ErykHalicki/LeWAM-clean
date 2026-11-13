"""Direction of the best-checkpoint comparison: `best_mode` decides whether a run is
selected on the lowest value (a loss) or the highest (a success rate)."""

from lewam.experiments.training.base import _improved


def test_min_mode_prefers_lower():
    assert _improved(0.4, 0.5, "min")
    assert not _improved(0.6, 0.5, "min")
    assert not _improved(0.5, 0.5, "min")


def test_max_mode_prefers_higher():
    assert _improved(0.6, 0.5, "max")
    assert not _improved(0.4, 0.5, "max")
    assert not _improved(0.5, 0.5, "max")


# training() rejects anything but min/max up front, so a typo'd best_mode
# never reaches _improved
