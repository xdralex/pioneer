import os
from dataclasses import dataclass
from time import sleep
from typing import Optional, Tuple, Dict, List

import numpy as np
from gym import spaces
from gym import utils
from gym.utils import seeding
from numpy.random.mtrand import RandomState

from collections_util import arr2str, dict2str
from pioneer.envs.bullet import BulletEnv, RenderConfig, SimulationConfig, Scene

Action = np.ndarray
Observation = np.ndarray


@dataclass
class PioneerConfig:
    max_v_to_r: float = 2       # seconds^-1
    max_a_to_v: float = 10      # seconds^-1

    done_distance: float = 0.1

    potential_k: float = 100.0
    potential_s: float = 25.0
    step_penalty: float = 1 / 125
    done_reward: float = 100.0


class PioneerEnv(BulletEnv[Action, Observation], utils.EzPickle):
    def __init__(self,
                 headless: bool = True,
                 pioneer_config: Optional[PioneerConfig] = None,
                 simulation_config: Optional[SimulationConfig] = None,
                 render_config: Optional[RenderConfig] = None):
        # initialization
        model_path = os.path.join(os.path.dirname(__file__), 'assets/pioneer_6dof.urdf')
        BulletEnv.__init__(self, model_path, headless, simulation_config, render_config)
        utils.EzPickle.__init__(self)

        self.config = pioneer_config or PioneerConfig()

        # kinematics & environment
        self.r_lo, self.r_hi = self.joint_limits()                          # position limits [r_lo, r_hi]
        self.v_max = self.config.max_v_to_r * (self.r_hi - self.r_lo)       # absolute velocity limit [-v_max, v_max]
        self.a_max = self.config.max_a_to_v * self.v_max                    # absolute acceleration limit [-a_max, a_max]

        self.dt = self.world.step_time                                      # time passing between consecutive steps
        self.eps = 1e-5                                                     # numerical stability factor

        self.a = np.zeros(self.dof, dtype=np.float32)                       # acceleration set on the previous step
        self.v = np.zeros(self.dof, dtype=np.float32)                       # velocity reached at the end of the previous step
        self.r = self.joint_positions()                                     # position reached at the end of the previous step
        self.potential: float = 0                                           # potential reached at the end of the previous step

        # spaces
        self.action_space = spaces.Box(-self.a_max, self.a_max, dtype=np.float32)
        self.observation_space = self.observation_to_space(self.observe())
        self.reward_range = (-float('inf'), float('inf'))

        # randomness
        self.np_random: Optional[RandomState] = None
        self.seed()

    def reinit(self):
        self.a = np.zeros(self.dof, dtype=np.float32)
        self.v = np.zeros(self.dof, dtype=np.float32)
        self.r = self.np_random.uniform(self.r_lo, self.r_hi)

        self.reset_joint_positions(self.r)

        self.potential = 0

        # pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        # target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)
        #
        # diff = target_coords - pointer_coords
        # distance = np.linalg.norm(diff)
        #
        # self.potential = self.compute_potential(distance)

    def seed(self, seed=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def act(self, action: Action, world_index: int, step_index: int) -> Tuple[float, bool, Dict]:
        # computing angular kinematics
        a0 = self.a
        v0 = self.v
        r0 = self.r

        v1 = np.zeros(self.dof, dtype=np.float32)
        r1 = np.zeros(self.dof, dtype=np.float32)

        for i in range(self.dof):
            v1[i] = v0[i] + a0[i] * self.dt     # velocity reached at the end of step
            dt_p1 = self.dt                     # acceleration phase time
            dt_p2 = 0                           # uniform motion phase time

            if v1[i] > self.v_max[i]:
                dt_p1 = np.clip((self.v_max[i] - v0[i]) / (a0[i] + self.eps), 0, self.dt)
                dt_p2 = self.dt - dt_p1
                v1[i] = self.v_max[i]
            elif v1[i] < -self.v_max[i]:
                dt_p1 = np.clip((-self.v_max[i] - v0[i]) / (a0[i] + self.eps), 0, self.dt)
                dt_p2 = self.dt - dt_p1
                v1[i] = -self.v_max[i]

            r1[i] = r0[i] + 0.5 * (v0[i] + v1[i]) * dt_p1 + v1[i] * dt_p2

        r1 = np.clip(r1, self.r_lo, self.r_hi)

        # making the step
        self.a = action
        self.v = v1
        self.r = r1

        self.reset_joint_positions(self.r)

        # issuing rewards
        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)

        diff = target_coords - pointer_coords
        distance = np.linalg.norm(diff)

        old_potential = self.potential
        self.potential = self.compute_potential(distance)

        done = distance < self.config.done_distance

        reward_potential = self.potential - old_potential
        reward_step = -self.config.step_penalty
        reward_done = self.config.done_reward if done else 0
        reward = reward_potential + reward_step + reward_done

        info = {
            'r_pot': f'{reward_potential:.3f}',
            'r_step': f'{reward_step:.3f}',
            'r_done': f'{reward_done:.3f}',
            'r': f'{reward:.3f}',

            # 'ptr': arr2str(pointer_coords),
            # 'target': arr2str(target_coords),
            # 'diff': arr2str(diff),
            'dist': f'{distance:.3f}',
            'pot': f'{self.potential:.3f}',

            'j_act': arr2str(action),
            'j_pos': arr2str(np.array([x.position() for x in self.scene.joints])),
            'j_lo': arr2str(np.array([x.lower_limit for x in self.scene.joints])),
            'j_hi': arr2str(np.array([x.upper_limit for x in self.scene.joints]))
        }

        print(f'#{world_index}/#{step_index} - {dict2str(info, delim="  ")}')

        self.world.step()
        return reward, done, info

    def observe(self) -> Observation:
        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)

        diff = target_coords - pointer_coords
        distance = np.linalg.norm(diff)

        return np.concatenate([
            self.r,
            self.r_lo,
            self.r_hi,

            self.r - self.r_lo,
            self.r_hi - self.r,

            np.cos(self.r), np.sin(self.r),
            np.cos(self.r_lo), np.sin(self.r_lo),
            np.cos(self.r_hi), np.sin(self.r_hi),

            pointer_coords,
            target_coords,
            diff,

            np.array([distance]),
            np.array([self.potential])
        ])

    @property
    def dof(self) -> int:
        return len(self.scene.joints)

    def joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        lower_limits = np.array([x.lower_limit for x in self.scene.joints], dtype=np.float32)
        upper_limits = np.array([x.upper_limit for x in self.scene.joints], dtype=np.float32)
        return lower_limits, upper_limits

    def joint_positions(self) -> np.ndarray:
        return np.array([x.position() for x in self.scene.joints])

    def reset_joint_positions(self, positions: np.ndarray):
        positions_list = list(positions)
        assert len(positions_list) == self.dof

        for position, joint in zip(positions_list, self.scene.joints):
            joint.reset_state(position)

    def compute_potential(self, distance: float) -> float:
        k = self.config.potential_k
        s = self.config.potential_s

        return k / (distance / s + 1)

    # def act_best_distance(self, action: Action) -> Tuple[float, bool, Dict]:
    #     pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
    #     target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)
    #     diff = target_coords - pointer_coords
    #     distance = np.linalg.norm(diff)
    #
    #     reward = 0
    #     if distance < self.best_distance:
    #         reward += self.potential(distance) - self.potential(self.best_distance)
    #         self.best_distance = distance
    #     reward -= self.config.step_penalty
    #
    #     done = distance < self.config.stop_distance
    #
    #     action_list = list(action)
    #     assert len(action_list) == self.dof
    #     for velocity, joint in zip(action_list, self.scene.joints):
    #         joint.control_velocity(velocity)
    #
    #     self.world.step()
    #     return reward, done, dict()

    # def act_position(self, action: Action) -> Tuple[float, bool, Dict]:
    #     pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
    #     target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)
    #     diff = target_coords - pointer_coords
    #     distance = np.linalg.norm(diff)
    #
    #     reward = -distance
    #     done = distance < self.config.stop_distance
    #
    #     action_list = list(action)
    #     assert len(action_list) == self.dof
    #     for position, joint in zip(action_list, self.scene.joints):
    #         joint.control_position(position)
    #
    #     self.world.step()
    #     return reward, done, dict()

    # @staticmethod
    # def joint_positions_space(scene: Scene) -> spaces.Space:
    #     lower_limits = np.array([x.lower_limit for x in scene.joints], dtype=np.float32)
    #     upper_limits = np.array([x.upper_limit for x in scene.joints], dtype=np.float32)
    #     return spaces.Box(low=lower_limits, high=upper_limits, dtype=np.float32)

    # def act_control(self, action: Action, world_index: int, step_index: int) -> Tuple[float, bool, Dict]:
    #     pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
    #     target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)
    #
    #     # print(f'act: {pointer_coords} - {target_coords}')
    #
    #     diff = target_coords - pointer_coords
    #     distance = np.linalg.norm(diff)
    #
    #     old_potential = self.potential
    #     self.potential = self.compute_potential(distance)
    #
    #     done = distance < self.config.stop_distance
    #
    #     reward_potential = self.potential - old_potential
    #     reward_step = -self.config.step_penalty
    #     reward_done = self.config.done_reward if done else 0
    #     reward = reward_potential + reward_step + reward_done
    #
    #     action_list = list(action)
    #     assert len(action_list) == self.dof
    #     for velocity, joint in zip(action_list, self.scene.joints):
    #         joint.control_velocity(velocity)
    #
    #     info = {
    #         'r_pot': f'{reward_potential:.3f}',
    #         'r_step': f'{reward_step:.3f}',
    #         'r_done': f'{reward_done:.3f}',
    #         'r': f'{reward:.3f}',
    #
    #         # 'ptr': arr2str(pointer_coords),
    #         # 'target': arr2str(target_coords),
    #         # 'diff': arr2str(diff),
    #         'dist': f'{distance:.3f}',
    #         'pot': f'{self.potential:.3f}',
    #
    #         'j_act': arr2str(action),
    #         'j_pos': arr2str(np.array([x.position() for x in self.scene.joints])),
    #         'j_lo': arr2str(np.array([x.lower_limit for x in self.scene.joints])),
    #         'j_hi': arr2str(np.array([x.upper_limit for x in self.scene.joints]))
    #     }
    #
    #     print(f'#{world_index}/#{step_index} - {dict2str(info, delim="  ")}')
    #
    #     self.world.step()
    #     return reward, done, info

    # @staticmethod
    # def joint_velocities_space(scene: Scene, velocity_factor: float) -> spaces.Space:
    #     lower_limits = np.array([x.lower_limit for x in scene.joints], dtype=np.float32)
    #     upper_limits = np.array([x.upper_limit for x in scene.joints], dtype=np.float32)
    #
    #     distance = upper_limits - lower_limits
    #     velocity = distance * velocity_factor
    #
    #     return spaces.Box(low=-velocity, high=velocity, dtype=np.float32)

    @staticmethod
    def observation_to_space(observation: Observation) -> spaces.Space:
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        return spaces.Box(low, high, dtype=observation.dtype)


if __name__ == '__main__':
    env = PioneerEnv(headless=False)
    env.reset_gui_camera()

    # print(env.scene)
    #
    # print(f'{env.action_space.low} - {env.action_space.high}')
    # lower_limits = np.array([x.lower_limit for x in env.scene.joints], dtype=np.float32)
    # upper_limits = np.array([x.upper_limit for x in env.scene.joints], dtype=np.float32)
    #
    # print(f'{lower_limits} - {upper_limits}')
    #
    # print(np.array(env.scene.items_by_name['robot:pointer'].pose().xyz))
    # print(np.array(env.scene.items_by_name['target'].pose().xyz))
    #
    # # print(f'{env.action_space.low} - {env.action_space.high}')
    #
    # # print(env.observation_space)
    #
    # # env.act(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    #
    # target_joint = env.scene.joints_by_name['robot:arm3_to_rotator3']
    # # target_joint.control_velocity(velocity=10.0)
    #
    # pos = target_joint.lower_limit
    # diff = target_joint.upper_limit - target_joint.lower_limit
    #
    # print(target_joint.upper_limit)
    # print(target_joint.lower_limit)
    #
    # target_joint.reset_state(0.0, 4.0)

    while True:
        # env.act(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), world_index=0, step_index=0)

        # rr = target_joint.upper_limit - target_joint.lower_limit
        # if target_joint.position() < target_joint.lower_limit + 0.01 * rr:
        #     target_joint.control_velocity(velocity=1.0)
        #
        # if target_joint.position() > target_joint.upper_limit - 0.01 * rr:
        #     target_joint.control_velocity(velocity=-1.0)

        # target_joint.control_velocity(velocity=10.0, max_force=1000)

        # target_joint.control_position(pos, position_gain=0.1, velocity_gain=0.1, max_force=0)
        # pos += diff / 96

        # print(f'{pos} - {target_joint.position()}')

        # print(f'{target_joint.position()}')

        # env.act(np.array([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0]), world_index=0, step_index=0)

        env.world.step()
        sleep(env.world.step_time)
