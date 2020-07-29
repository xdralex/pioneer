import os
from dataclasses import dataclass
from time import sleep
from typing import Optional, Tuple, Dict, List

import numpy as np
from gym import spaces
from gym import utils
from gym.utils import seeding
from numpy.random.mtrand import RandomState

from pioneer.collections_util import arr2str
from pioneer.envs.bullet import BulletEnv, RenderConfig, SimulationConfig

Action = np.ndarray
Observation = np.ndarray


@dataclass
class PioneerKinematicConfig:
    max_v_to_r: float = 2       # seconds^-1
    max_a_to_v: float = 10      # seconds^-1

    done_distance: float = 0.1

    potential_k: float = 100.0
    potential_s: float = 25.0
    step_penalty: float = 1 / 125
    done_reward: float = 100.0

    target_lo: Tuple[float, float, float] = (10, -8, 1)
    target_hi: Tuple[float, float, float] = (25, 8, 7)
    target_radius: float = 0.2
    target_rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.5)


# TODO: inheritance is hard – separation between this and BulletEnv should be much cleaner
class PioneerKinematicEnv(BulletEnv[Action, Observation], utils.EzPickle):
    def __init__(self,
                 headless: bool = True,
                 pioneer_config: Optional[PioneerKinematicConfig] = None,
                 simulation_config: Optional[SimulationConfig] = None,
                 render_config: Optional[RenderConfig] = None):
        # randomness
        self.np_random: Optional[RandomState] = None
        self.seed()

        # initialization
        model_path = os.path.join(os.path.dirname(__file__), 'assets/pioneer_knm_6dof.urdf')
        BulletEnv.__init__(self, model_path, headless, simulation_config, render_config)
        utils.EzPickle.__init__(self)

        self.config = pioneer_config or PioneerKinematicConfig()

        # kinematics & environment
        self.r_lo, self.r_hi = self.joint_limits()                          # position limits [r_lo, r_hi]
        self.v_max = self.config.max_v_to_r * (self.r_hi - self.r_lo)       # absolute velocity limit [-v_max, v_max]
        self.a_max = self.config.max_a_to_v * self.v_max                    # absolute acceleration limit [-a_max, a_max]

        self.dt = self.world.step_time                                      # time passing between consecutive steps
        self.eps = 1e-5                                                     # numerical stability factor

        self.a: Optional[np.ndarray] = None                                 # acceleration set on the previous step
        self.v: Optional[np.ndarray] = None                                 # velocity reached at the end of the previous step
        self.r: Optional[np.ndarray] = None                                 # position reached at the end of the previous step
        self.potential: Optional[float] = None                              # potential reached at the end of the previous step

        # final touches
        self.reset_world()

        # spaces
        self.action_space = spaces.Box(-self.a_max, self.a_max, dtype=np.float32)
        self.observation_space = self.observation_to_space(self.observe())
        self.reward_range = (-float('inf'), float('inf'))

    def reset_world(self,
                    joint_positions: Optional[np.ndarray] = None,
                    target_position: Optional[Tuple[float, float, float]] = None):

        if joint_positions is None:
            joint_positions = self.np_random.uniform(self.r_lo, self.r_hi)

        if target_position is None:
            assert len(self.config.target_lo) == 3
            assert len(self.config.target_hi) == 3

            target_lo = np.array(self.config.target_lo)
            target_hi = np.array(self.config.target_hi)

            target_position = tuple(self.np_random.uniform(target_lo, target_hi))

        self.a = np.zeros(self.dof, dtype=np.float32)
        self.v = np.zeros(self.dof, dtype=np.float32)
        self.r = joint_positions

        self.reset_joint_positions(self.r)
        self.scene.create_body_sphere('target',
                                      collision=False,
                                      mass=0.0,
                                      radius=self.config.target_radius,
                                      position=target_position,
                                      orientation=self.scene.rpy2quat((0, 0, 0)),
                                      rgba_color=self.config.target_rgba)

        self.potential = 0

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
            v1[i] = v0[i] + a0[i] * self.dt  # velocity reached at the end of step
            dt_p1 = self.dt  # acceleration phase time
            dt_p2 = 0  # uniform motion phase time

            if v1[i] > self.v_max[i]:
                dt_p1 = np.clip((self.v_max[i] - v0[i]) / (a0[i] + self.eps), 0, self.dt)
                dt_p2 = self.dt - dt_p1
                v1[i] = self.v_max[i]
            elif v1[i] < -self.v_max[i]:
                dt_p1 = np.clip((-self.v_max[i] - v0[i]) / (a0[i] + self.eps), 0, self.dt)
                dt_p2 = self.dt - dt_p1
                v1[i] = -self.v_max[i]

            r1[i] = r0[i] + 0.5 * (v0[i] + v1[i]) * dt_p1 + v1[i] * dt_p2
            if r1[i] >= self.r_hi[i]:
                r1[i] = self.r_hi[i]
                v1[i] = 0

            if r1[i] <= self.r_lo[i]:
                r1[i] = self.r_lo[i]
                v1[i] = 0

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
            'rw': f'{reward:.3f}',

            'dist': f'{distance:.3f}',
            'pot': f'{self.potential:.3f}',

            'a': arr2str(self.a),
            'v': arr2str(self.v),
            'r': arr2str(self.r)
        }

        # print(f'#{world_index}/#{step_index} - {dict2str(info, delim="  ")}')

        self.world.step()
        return reward, done, info

    def observe(self) -> Observation:
        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)

        diff = target_coords - pointer_coords
        distance = np.linalg.norm(diff)

        r_lo_dist = self.r - self.r_lo
        r_hi_diff = self.r_hi - self.r

        return np.concatenate([
            self.r, np.cos(self.r), np.sin(self.r),
            self.r_lo, np.cos(self.r_lo), np.sin(self.r_lo),
            self.r_hi, np.cos(self.r_hi), np.sin(self.r_hi),

            r_lo_dist, np.cos(r_lo_dist), np.sin(r_lo_dist),
            r_hi_diff, np.cos(r_hi_diff), np.sin(r_hi_diff),

            self.v, np.cos(self.v), np.sin(self.v),
            self.a, np.cos(self.a), np.sin(self.a),

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

    @staticmethod
    def observation_to_space(observation: Observation) -> spaces.Space:
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        return spaces.Box(low, high, dtype=observation.dtype)


if __name__ == '__main__':
    env = PioneerKinematicEnv(headless=False)
    env.reset_gui_camera()

    # env.scene.create_body_sphere('test',
    #                              collision=False,
    #                              mass=0.0,
    #                              radius=0.2,
    #                              position=(15.0, 3.0, 2.0),
    #                              orientation=env.scene.rpy2quat((0, 0, 0)),
    #                              rgba_color=(0.1, 0.9, 0.1, 0.5))
    #
    # env.scene.items_by_name['test'].reset_pose((30.0, 3.0, 2.0), env.scene.rpy2quat((0, 0, 0)))

    # target_joint = env.scene.joints_by_name['robot:arm3_to_rotator3']

    count = 0
    while True:
        # rr = target_joint.upper_limit - target_joint.lower_limit
        # if target_joint.position() < target_joint.lower_limit + 0.01 * rr:
        #     target_joint.control_velocity(velocity=1.0)
        #
        # if target_joint.position() > target_joint.upper_limit - 0.01 * rr:
        #     target_joint.control_velocity(velocity=-1.0)
        # target_joint.control_velocity(velocity=10.0, max_force=1000)
        # target_joint.control_position(pos, position_gain=0.1, velocity_gain=0.1, max_force=0)

        env.world.step()
        sleep(env.world.step_time)
