from .base import BaseExperiment
from .eval.base import EvalMixin
from .training.base import TrainingMixin


class PushTExperiment(TrainingMixin, EvalMixin, BaseExperiment):
    """Trains the flow-matching policy on swm PushT demonstrations, then evaluates
    success rate via rollouts in the same swm/PushT-v1 world.
    """
