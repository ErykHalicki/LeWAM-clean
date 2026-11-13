"""LeWAM-local env variants, registered under gym ids on import."""

from gymnasium.envs import registration


def _register(id, entry_point):
    if id not in registration.registry:
        registration.register(id=id, entry_point=entry_point)


_register(
    "swm/PushTFixedGoal-v1",
    "lewam.envs.pusht:PushTFixedGoal",
)

_register(
    "swm/PushTFixedGoalAbsoluteAction-v1",
    "lewam.envs.pusht:PushTFixedGoalAbsoluteAction",
)

_register(
    "swm/PushTAbsoluteAction-v1",
    "lewam.envs.pusht:PushTAbsoluteAction",
)

_register(
    "swm/TwoRoomGoalImage-v1",
    "lewam.envs.two_room:TwoRoomGoalImageEnv",
)
