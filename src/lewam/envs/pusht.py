"""PushT with the classic fixed goal and block-only success.

The stock swm PushT env draws its goal outline from the variation space (center T at
(256, 256), angle pi/4 by default) but scores success against `goal_state`, a fresh
random agent+block sample taken at every reset -- a hidden target unrelated to the
drawn goal. Its `eval_state` also requires the AGENT position to sit inside the same
20px ball as the block. A policy pushing the block to the visible goal therefore
never terminates successfully.

This subclass pins `goal_state` to the drawn default goal and scores success on the
block pose alone, matching the classic PushT task: block within 20px of (256, 256)
and within 20 degrees of pi/4, agent anywhere.
"""

import numpy as np
from gymnasium import spaces

from stable_worldmodel.envs.pusht.env import PushT

# [agent_x, agent_y, block_x, block_y, block_angle, vel_x, vel_y]; the agent
# entry only affects the rendered goal image, never success
FIXED_GOAL_STATE = np.array([256.0, 400.0, 256.0, 256.0, np.pi / 4, 0.0, 0.0])


class PushTFixedGoal(PushT):
    def reset(self, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("goal_state", FIXED_GOAL_STATE.copy())
        return super().reset(seed=seed, options=options)

    def eval_state(self, goal_state, cur_state):
        # angle compared on the circle, so accumulated full rotations of the block
        # do not count against it
        pos_diff = np.linalg.norm(goal_state[2:4] - cur_state[2:4])
        angle_diff = np.abs(
            (goal_state[4] - cur_state[4] + np.pi) % (2.0 * np.pi) - np.pi
        )
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = float(np.linalg.norm(goal_state[2:5] - cur_state[2:5]))
        return success, state_dist


class PushTAbsoluteAction(PushT):
    """Stock swm PushT taking ABSOLUTE pixel target actions in [0, window_size] instead
    of swm's default relative [-1, 1] delta (`relative=False` makes step() use the action
    as the PD controller's target position directly). This matches the gym-pusht convention
    `lerobot/pusht` was recorded with, so a policy trained on that data evaluates in the swm
    env without an action-space conversion.

    Goal handling and the stock `eval_state` success metric (agent AND block jointly
    within 20px / 20deg of the goal state) are untouched. For the classic fixed-goal
    task use the subclass below.
    """

    def __init__(self, *args, **kwargs):
        kwargs["relative"] = False
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(
            low=0.0, high=float(self.window_size), shape=(2,), dtype=np.float32
        )


class PushTFixedGoalAbsoluteAction(PushTFixedGoal, PushTAbsoluteAction):
    """PushTFixedGoal (pinned goal, block-only success) with absolute pixel actions.
    Rendering/goal are still swm's, so an image-appearance gap versus the recorded
    frames may remain.
    """
