from .base import BaseExperiment
from .eval.base import EvalMixin
from .training.base import TrainingMixin


class TwoRoomExperiment(TrainingMixin, EvalMixin, BaseExperiment):
    """Trains the flow-matching policy on swm TwoRoom demonstrations, then evaluates
    success rate via rollouts in the same swm/TwoRoom-v1 world.
    """
