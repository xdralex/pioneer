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
class PioneerConfig:
    max_v_to_r: float = 2       # seconds^-1
    max_a_to_v: float = 10      # seconds^-1

    done_distance: float = 0.1

    award_max: float = 100.0
    award_done: float = 5.0
    award_potential_slope: float = 10.0
    penalty_step: float = 1 / 100

    target_lo: Tuple[float, float, float] = (15, -10, 2)
    target_hi: Tuple[float, float, float] = (25, 10, 6)
    target_radius: float = 0.01
    target_rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    target_aim_radius: float = 0.2
    target_aim_rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.5)

    obstacle_size_lo: Tuple[float, float, float] = (0.1, 0.1, 5)
    obstacle_size_hi: Tuple[float, float, float] = (2, 2, 10)
    obstacle_pos_lo: Tuple[float, float] = (5, -10)
    obstacle_pos_hi: Tuple[float, float] = (20, 10)


# TODO: inheritance is hard – separation between this and BulletEnv should be much cleaner
class PioneerEnv(BulletEnv[Action, Observation], utils.EzPickle):
    def __init__(self,
                 headless: bool = True,
                 pioneer_config: Optional[PioneerConfig] = None,
                 simulation_config: Optional[SimulationConfig] = None,
                 render_config: Optional[RenderConfig] = None):
        # randomness
        self.np_random: Optional[RandomState] = None
        self.seed()

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

        self.potential: Optional[float] = None                              # potential reached at the end of the previous step

        # obstacles
        self.obstacle_size: Optional[np.ndarray] = None
        self.obstacle_position: Optional[np.ndarray] = None

        # final touches
        self.reset_world()

        # spaces
        self.action_space = spaces.Box(-self.a_max, self.a_max, dtype=np.float32)
        self.observation_space = self.observation_to_space(self.observe())
        self.reward_range = (-float('inf'), float('inf'))

    def reset_world(self):

        assert len(self.config.target_lo) == 3
        assert len(self.config.target_hi) == 3

        assert len(self.config.obstacle_size_lo) == 3
        assert len(self.config.obstacle_size_hi) == 3
        assert len(self.config.obstacle_pos_lo) == 2
        assert len(self.config.obstacle_pos_hi) == 2

        joint_positions = self.np_random.uniform(self.r_lo, self.r_hi)
        target_position = tuple(self.np_random.uniform(np.array(self.config.target_lo), np.array(self.config.target_hi)))
        obstacle_size = tuple(self.np_random.uniform(np.array(self.config.obstacle_size_lo), np.array(self.config.obstacle_size_hi)))
        obstacle_pos = tuple(self.np_random.uniform(np.array(self.config.obstacle_pos_lo), np.array(self.config.obstacle_pos_hi)))

        self.obstacle_size = np.array(obstacle_size, dtype=np.float32)
        self.obstacle_position = np.array(obstacle_pos, dtype=np.float32)

        self.reset_joint_states(joint_positions)

        self.scene.create_body_sphere('target',
                                      collision=True,
                                      mass=0.0,
                                      radius=self.config.target_radius,
                                      position=target_position,
                                      orientation=self.scene.rpy2quat((0, 0, 0)),
                                      rgba_color=self.config.target_rgba)

        self.scene.create_body_sphere('target:aim',
                                      collision=True,
                                      mass=0.0,
                                      radius=self.config.target_aim_radius,
                                      position=target_position,
                                      orientation=self.scene.rpy2quat((0, 0, 0)),
                                      rgba_color=self.config.target_aim_rgba)

        self.scene.create_body_box(name='obstacle',
                                   collision=True,
                                   mass=0.0,
                                   half_extents=obstacle_size,
                                   position=(obstacle_pos[0], obstacle_pos[1], obstacle_size[2]),
                                   orientation=self.scene.rpy2quat((0, 0, 0)),
                                   rgba_color=(0, 0, 0, 1))

        self.potential = 0

    def seed(self, seed=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def act(self, action: Action, world_index: int, step_index: int) -> Tuple[float, bool, Dict]:
        # computing angular kinematics
        a = action
        r = self.joint_positions()

        v0 = self.joint_velocities()
        v1 = v0 + a * self.dt
        v1 = np.clip(v1, -self.v_max, self.v_max)

        self.reset_joint_states(r, v1)

        # issuing rewards
        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)

        diff = target_coords - pointer_coords
        distance = np.linalg.norm(diff)

        old_potential = self.potential
        self.potential = self.compute_potential(distance)

        done = distance < self.config.done_distance

        reward_potential = self.potential - old_potential
        reward_step = -self.config.penalty_step
        reward_done = self.config.award_done if done else 0
        reward = reward_potential + reward_step + reward_done

        info = {
            'rew_pot': f'{reward_potential:.3f}',
            'rew_step': f'{reward_step:.3f}',
            'rew_done': f'{reward_done:.3f}',
            'rew': f'{reward:.3f}',

            'dist': f'{distance:.3f}',
            'pot': f'{self.potential:.3f}',

            'a': arr2str(a),
            'v0': arr2str(v0),
            'v1': arr2str(v1),
            'r': arr2str(r)
        }

        self.world.step()
        return reward, done, info

    def observe(self) -> Observation:
        r = self.joint_positions()
        v = self.joint_velocities()

        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)

        diff = target_coords - pointer_coords
        distance = np.linalg.norm(diff)

        r_lo_dist = r - self.r_lo
        r_hi_diff = self.r_hi - r

        return np.concatenate([
            r, np.cos(r), np.sin(r),
            v, np.cos(v), np.sin(v),

            self.r_lo, np.cos(self.r_lo), np.sin(self.r_lo),
            self.r_hi, np.cos(self.r_hi), np.sin(self.r_hi),

            r_lo_dist, np.cos(r_lo_dist), np.sin(r_lo_dist),
            r_hi_diff, np.cos(r_hi_diff), np.sin(r_hi_diff),

            self.obstacle_size,
            self.obstacle_position,

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

    def joint_velocities(self) -> np.ndarray:
        return np.array([x.velocity() for x in self.scene.joints])

    def reset_joint_states(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None):
        positions_list = list(positions)
        velocities_list = [None] * self.dof if velocities is None else list(velocities)

        assert len(positions_list) == self.dof
        assert len(velocities_list) == self.dof

        for position, velocity, joint in zip(positions_list, velocities_list, self.scene.joints):
            joint.reset_state(position, velocity)

    def compute_potential(self, distance: float) -> float:
        m = self.config.award_max - self.config.award_done      # max potential (achieved when the distance is 0)
        s = self.config.award_potential_slope                   # slope of the potential curve

        return m / (distance / s + 1)

    @staticmethod
    def observation_to_space(observation: Observation) -> spaces.Space:
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        return spaces.Box(low, high, dtype=observation.dtype)


if __name__ == '__main__':
    env = PioneerEnv(headless=False)
    env.reset_gui_camera()

    # env.scene.create_body_box(name='obstacle:1',
    #                           collision=True,
    #                           mass=0.0,
    #                           half_extents=(0.5, 0.5, 5.0),
    #                           position=(10, 5, 0),
    #                           orientation=env.scene.rpy2quat((0, 0, 0)),
    #                           rgba_color=(0, 0, 0, 1))

    # env.scene.create_body_sphere(name='test',
    #                              collision=False,
    #                              mass=0.0,
    #                              radius=0.2,
    #                              position=(15.0, 3.0, 2.0),
    #                              orientation=env.scene.rpy2quat((0, 0, 0)),
    #                              rgba_color=(0.1, 0.9, 0.1, 0.5))
    #
    # env.scene.items_by_name['test'].reset_pose((30.0, 3.0, 2.0), env.scene.rpy2quat((0, 0, 0)))

    target_joint = env.scene.joints_by_name['robot:hinge1_to_arm1']
    # target_joint.control_velocity(velocity=1.0)
    env_velocity = 1.0

    count = 0
    while True:
        print(env.bullet.getContactPoints())

        target_joint.reset_state(target_joint.position(), velocity=env_velocity)

        # print(f'{target_joint.position()} - {target_joint.lower_limit}..{target_joint.upper_limit}')

        rr = target_joint.upper_limit - target_joint.lower_limit
        if target_joint.position() < target_joint.lower_limit + 0.01 * rr:
            # target_joint.control_velocity(velocity=1.0)
            env_velocity = 1.0

        if target_joint.position() > target_joint.upper_limit - 0.01 * rr:
            # target_joint.control_velocity(velocity=-1.0)
            env_velocity = -1.0

        # target_joint.control_velocity(velocity=10.0, max_force=1000)
        # target_joint.control_position(pos, position_gain=0.1, velocity_gain=0.1, max_force=0)

        env.world.step()
        sleep(env.world.step_time)
