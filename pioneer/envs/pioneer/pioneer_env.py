import math
import os
from dataclasses import dataclass
from time import sleep
from typing import Optional, Tuple, Dict, List

import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import utils

from pioneer.envs.bullet import BulletEnv, RenderConfig, SimulationConfig, Scene

Action = np.ndarray
Observation = np.ndarray


@dataclass
class PioneerConfig:
    velocity_factor: float = 0.2
    potential_scale: float = 5.0
    step_penalty: float = 1 / 125
    stop_distance: float = 0.1


class PioneerEnv(BulletEnv[Action, Observation], utils.EzPickle):
    def __init__(self,
                 headless: bool = True,
                 pioneer_config: Optional[PioneerConfig] = None,
                 simulation_config: Optional[SimulationConfig] = None,
                 render_config: Optional[RenderConfig] = None):

        model_path = os.path.join(os.path.dirname(__file__), 'assets/pioneer_6dof.urdf')
        BulletEnv.__init__(self, model_path, headless, simulation_config, render_config)
        utils.EzPickle.__init__(self)

        self.config = pioneer_config or PioneerConfig()

        # TODO: self.action_space = self.joint_velocities_space(self.scene, self.config.velocity_factor)
        self.action_space = self.joint_positions_space(self.scene)
        self.observation_space = self.observation_to_space(self.observe())
        self.reward_range = (-float('inf'), float('inf'))

        self.np_random = None
        self.seed()

        self.best_distance = math.inf

    def seed(self, seed=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def act(self, action: Action) -> Tuple[float, bool, Dict]:
        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)
        diff = target_coords - pointer_coords
        distance = np.linalg.norm(diff)

        reward = 0
        if distance < self.best_distance:
            reward += self.potential(distance) - self.potential(self.best_distance)
            self.best_distance = distance
        reward -= self.config.step_penalty

        done = distance < self.config.stop_distance

        action_list = list(action)
        assert len(action_list) == len(self.scene.joints)
        for velocity, joint in zip(action_list, self.scene.joints):
            joint.control_velocity(velocity)

        self.world.step()
        return reward, done, dict()

    # def act(self, action: Action) -> Tuple[float, bool, Dict]:
    #     pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
    #     target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)
    #     diff = target_coords - pointer_coords
    #     distance = np.linalg.norm(diff)
    #
    #     reward = -distance
    #     done = distance < self.config.stop_distance
    #
    #     action_list = list(action)
    #     assert len(action_list) == len(self.scene.joints)
    #     for position, joint in zip(action_list, self.scene.joints):
    #         joint.control_position(position)
    #
    #     self.world.step()
    #     return reward, done, dict()

    def observe(self) -> Observation:
        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)
        diff = target_coords - pointer_coords
        distance = np.linalg.norm(diff)

        joint_positions = np.array([x.position() for x in self.scene.joints])
        joint_lower_limits = np.array([x.lower_limit for x in self.scene.joints])
        joint_upper_limits = np.array([x.upper_limit for x in self.scene.joints])

        return np.concatenate([
            joint_positions,
            joint_lower_limits,
            joint_upper_limits,

            joint_positions - joint_lower_limits,
            joint_upper_limits - joint_positions,

            np.cos(joint_positions), np.sin(joint_positions),
            np.cos(joint_lower_limits), np.sin(joint_lower_limits),
            np.cos(joint_upper_limits), np.sin(joint_upper_limits),

            pointer_coords,
            target_coords,
            diff,

            np.array([distance])
        ])

    def potential(self, distance: float) -> float:
        return 1 / (distance + 1 / self.config.potential_scale)

    @staticmethod
    def joint_positions_space(scene: Scene) -> spaces.Space:
        lower_limits = np.array([x.lower_limit for x in scene.joints], dtype=np.float32)
        upper_limits = np.array([x.upper_limit for x in scene.joints], dtype=np.float32)
        return spaces.Box(low=lower_limits, high=upper_limits, dtype=np.float32)

    @staticmethod
    def joint_velocities_space(scene: Scene, velocity_factor: float) -> spaces.Space:
        lower_limits = np.array([x.lower_limit for x in scene.joints], dtype=np.float32)
        upper_limits = np.array([x.upper_limit for x in scene.joints], dtype=np.float32)

        distance = upper_limits - lower_limits
        velocity = distance * velocity_factor

        return spaces.Box(low=-velocity, high=velocity, dtype=np.float32)

    @staticmethod
    def observation_to_space(observation: Observation) -> spaces.Space:
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        return spaces.Box(low, high, dtype=observation.dtype)


if __name__ == '__main__':
    env = PioneerEnv(headless=False)
    env.reset_gui_camera()

    env.act(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    # target_joint = env.scene.joints_by_name['robot:arm3_to_rotator3']
    # target_joint.control_velocity(velocity=1.0)

    while True:
        # rr = target_joint.upper_limit - target_joint.lower_limit
        # if target_joint.position() < target_joint.lower_limit + 0.01 * rr:
        #     target_joint.control_velocity(velocity=1.0)
        #
        # if target_joint.position() > target_joint.upper_limit - 0.01 * rr:
        #     target_joint.control_velocity(velocity=-1.0)

        env.world.step()
        sleep(env.world.timestep)
