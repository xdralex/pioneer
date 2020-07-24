import os
from time import sleep
from typing import Optional, Tuple, Dict, List

import numpy as np
from gym import spaces
from gym.utils import seeding

from pioneer.envs.bullet import BulletEnv, RenderConfig, SimulationConfig

Action = np.ndarray
Observation = np.ndarray


class PioneerEnv(BulletEnv[Action, Observation]):
    def __init__(self,
                 headless: bool = True,
                 simulation_config: Optional[SimulationConfig] = None,
                 render_config: Optional[RenderConfig] = None):

        model_path = os.path.join(os.path.dirname(__file__), 'assets/pioneer_6dof.urdf')
        BulletEnv.__init__(self, model_path, headless, simulation_config, render_config)

        self.action_space = None
        self.observation_space = self.convert_observation_to_space(self.observe())
        self.reward_range = (-float('inf'), float('inf'))

        self.np_random = None

        self.seed()

    def seed(self, seed=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def act(self, action: Action) -> Tuple[float, bool, Dict]:
        pass

    def observe(self) -> Observation:
        pass

    @staticmethod
    def convert_observation_to_space(observation: Observation) -> spaces.Space:
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        return spaces.Box(low, high, dtype=observation.dtype)


if __name__ == '__main__':
    env = PioneerEnv(headless=False)
    env.reset_gui_camera()

    print(env.scene)

    target_joint = env.scene.joints_by_name['robot:arm3_to_rotator3']
    target_joint.control_velocity(velocity=1.0)

    while True:
        rr = target_joint.upper_limit - target_joint.lower_limit
        if target_joint.position() < target_joint.lower_limit + 0.01 * rr:
            target_joint.control_velocity(velocity=1.0)

        if target_joint.position() > target_joint.upper_limit - 0.01 * rr:
            target_joint.control_velocity(velocity=-1.0)

        env.world.step()
        sleep(env.world.timestep)
