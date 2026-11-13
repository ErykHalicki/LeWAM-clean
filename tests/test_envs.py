import numpy as np
import pytest

pytest.importorskip("stable_worldmodel")

import gymnasium as gym  # noqa: E402

import lewam.envs  # noqa: E402, F401  (registers LeWAM env variants)

from lewam.envs.pusht import FIXED_GOAL_STATE  # noqa: E402


def test_pusht_fixed_goal_pins_goal_state_across_resets():
    env = gym.make("swm/PushTFixedGoal-v1").unwrapped
    for seed in (0, 1, 2):
        env.reset(seed=seed)
        assert np.allclose(env.goal_state, FIXED_GOAL_STATE)


def test_pusht_fixed_goal_success_ignores_agent():
    env = gym.make("swm/PushTFixedGoal-v1").unwrapped
    env.reset(seed=0)
    goal = env.goal_state

    ok, _ = env.eval_state(goal, np.array([50.0, 50.0, 260.0, 252.0, np.pi / 4, 0, 0]))
    assert ok
    ok, _ = env.eval_state(goal, np.array([256.0, 400.0, 300.0, 256.0, np.pi / 4, 0, 0]))
    assert not ok
    ok, _ = env.eval_state(
        goal, np.array([0.0, 0.0, 256.0, 256.0, np.pi / 4 + 2 * np.pi, 0, 0])
    )
    assert ok
    ok, _ = env.eval_state(
        goal, np.array([0.0, 0.0, 256.0, 256.0, np.pi / 4 + np.pi / 4, 0, 0])
    )
    assert not ok


def test_pusht_absolute_action_uses_pixel_action_space():
    # lerobot/pusht actions are absolute [0, window] pixels, not swm's relative [-1, 1]
    env = gym.make("swm/PushTFixedGoalAbsoluteAction-v1").unwrapped
    assert env.relative is False
    assert float(env.action_space.low[0]) == 0.0
    assert float(env.action_space.high[0]) == float(env.window_size)
    env.reset(seed=0)
    assert np.allclose(env.goal_state, FIXED_GOAL_STATE)


def test_two_room_goal_image_emits_goal_frame():
    env = gym.make("swm/TwoRoomGoalImage-v1").unwrapped
    _, info = env.reset(seed=0)
    assert "goal" in info
    goal = np.asarray(info["goal"])
    assert goal.ndim == 3 and goal.shape[-1] == 3 and goal.dtype == np.uint8
