"""TwoRoom variant that emits a goal image, for latent world-model (encoded_goal) eval.

The stock swm TwoRoom env reports only the numeric goal (`target_position`) in its info, so
JEPA-style planning has no goal image to encode, unlike PushT, which renders `self._goal`.
This subclass renders the goal as the scene with the agent at the target position and adds it
to `info['goal']` (the World resizes it to the eval `image_shape`), so the CEM cost can
compare the rolled-out latent to the encoded goal.
"""

from stable_worldmodel.envs.two_room.env import TwoRoomEnv


class TwoRoomGoalImageEnv(TwoRoomEnv):
    def _get_info(self):
        info = super()._get_info()
        goal_chw = self._render_frame(agent_pos=self.target_position)
        info["goal"] = goal_chw.cpu().numpy().transpose(1, 2, 0)
        return info
