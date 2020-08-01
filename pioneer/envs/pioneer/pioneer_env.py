import os
from dataclasses import dataclass
from time import sleep
from typing import Optional, Tuple, Dict, List

import numpy as np
from gym import spaces
from gym import utils
from gym.utils import seeding
from numpy.random.mtrand import RandomState

from pioneer.collections_util import arr2str, dict2str
from pioneer.envs.bullet import BulletEnv, RenderConfig, SimulationConfig

Action = np.ndarray
Observation = np.ndarray


@dataclass
class PioneerConfig:
    r_to_max_v: float = 2.0                                                         # sec

    done_distance: float = 0.01                                                     # units

    award_max: float = 100.0                                                        # $
    award_done: float = 5.0                                                         # $
    award_potential_slope: float = 10.0
    penalty_step: float = 1 / 100                                                   # $

    target_lo: Tuple[float, float, float] = (1.5, -1.0, 0.2)                        # units
    target_hi: Tuple[float, float, float] = (2.5, 1.0, 0.6)                         # units
    target_aim_radius: float = 0.001                                                # units
    target_aim_rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    target_halo_radius: float = 0.025                                               # units
    target_halo_rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.5)

    obstacle_size_lo: Tuple[float, float, float] = (0.05, 0.05, 0.3)                # units
    obstacle_size_hi: Tuple[float, float, float] = (0.2, 0.2, 1)                    # units
    obstacle_pos_lo: Tuple[float, float] = (1, -1)                                  # units
    obstacle_pos_hi: Tuple[float, float] = (2, 1)                                   # units


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
        self.r_lo, self.r_hi = self.joint_limits()                          # position limits [r_lo, r_hi] (radians)
        self.v_max = (self.r_hi - self.r_lo) / self.config.r_to_max_v       # absolute velocity limit [-v_max, v_max] (radian/sec)
        self.dt = self.world.step_time                                      # time passing between consecutive steps (sec)

        # reached/set at the end of an action
        self.potential: Optional[float] = 0                                 # ($)

        # obstacles
        self.obstacle_size: Optional[np.ndarray] = None
        self.obstacle_position: Optional[np.ndarray] = None

        # final touches
        self.reset_world()

        # spaces
        v_norm = np.ones(self.dof, dtype=np.float32)
        self.action_space = spaces.Box(-v_norm, v_norm, dtype=np.float32)
        self.observation_space = self.observation_to_space(self.observe())
        self.reward_range = (-float('inf'), float('inf'))

    def reset_world(self) -> bool:

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

        self.scene.create_body_sphere('target:aim',
                                      collision=True,
                                      mass=0.0,
                                      radius=self.config.target_aim_radius,
                                      position=target_position,
                                      orientation=self.scene.rpy2quat((0, 0, 0)),
                                      rgba_color=self.config.target_aim_rgba)

        self.scene.create_body_sphere('target:halo',
                                      collision=False,
                                      mass=0.0,
                                      radius=self.config.target_halo_radius,
                                      position=target_position,
                                      orientation=self.scene.rpy2quat((0, 0, 0)),
                                      rgba_color=self.config.target_halo_rgba)

        self.scene.create_body_box(name='obstacle:1',
                                   collision=True,
                                   mass=0.0,
                                   half_extents=obstacle_size,
                                   position=(obstacle_pos[0], obstacle_pos[1], obstacle_size[2]),
                                   orientation=self.scene.rpy2quat((0, 0, 0)),
                                   rgba_color=(0, 0, 0, 1))

        self.potential = 0
        self.reset_joint_states(joint_positions)
        self.world.step()
        return not self.unwanted_collisions_present()

    def seed(self, seed=None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def act(self, action: Action, world_index: int, step_index: int) -> Tuple[float, bool, Dict]:
        # computing angular kinematics
        v = np.clip(action * self.v_max, -self.v_max, self.v_max)

        r0 = self.joint_positions()
        for _ in range(self.world.frame_skip):
            r = self.joint_positions()
            self.reset_joint_states(r, v)
            self.world.step()
        r1 = self.joint_positions()

        # issuing rewards
        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target:aim'].pose().xyz)

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

            'target': arr2str(target_coords),
            'pointer': arr2str(pointer_coords),

            'action': arr2str(action),
            'r0': arr2str(r0),
            'r1': arr2str(r1),
            'v': arr2str(v)
        }

        return reward, done, info

    def observe(self) -> Observation:
        r = self.joint_positions()

        pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        target_coords = np.array(self.scene.items_by_name['target:aim'].pose().xyz)

        diff = target_coords - pointer_coords
        distance = np.linalg.norm(diff)

        r_lo_dist = r - self.r_lo
        r_hi_diff = self.r_hi - r

        return np.concatenate([
            r, np.cos(r), np.sin(r),

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

    def unwanted_collisions_present(self) -> bool:
        contacts = self.contacts()

        allowed_robot = {
            'robot:rotator1': ['ground', 'robot:hinge1'],
            'robot:hinge1': ['robot:arm1', 'robot:rotator1'],
            'robot:arm1': ['robot:hinge1', 'robot:arm2'],
            'robot:arm2': ['robot:rotator2', 'robot:arm1'],
            'robot:rotator2': ['robot:hinge2', 'robot:arm2'],
            'robot:hinge2': ['robot:arm3', 'robot:rotator2'],
            'robot:arm3': ['robot:hinge2', 'robot:rotator3'],
            'robot:rotator3': ['robot:arm3', 'robot:effector'],
            'robot:effector': ['robot:pointer', 'robot:rotator3'],
            'robot:pointer': ['robot:effector']
        }

        def not_in_allowed(ls: List[str], allowed: List[str]):
            return [x for x in ls if x not in allowed]

        for k, vs in contacts.items():
            if k == 'ground':
                for v in vs:
                    if v not in ['robot:rotator1'] and not v.startswith('obstacle:'):
                        return True

            if k.startswith('obstacle:') and len(not_in_allowed(vs, ['ground'])) > 0:
                return True

            if k.startswith('target:') and len(vs) > 0:
                return True

            if k.startswith('robot:'):
                if len(not_in_allowed(vs, allowed_robot.get(k, []))) > 0:
                    return True

        return False

    def contacts(self) -> Dict[str, List[str]]:
        pairs = self.scene.pull_contacting_items()

        index = {}
        for a, b in pairs:
            index.setdefault(a, list()).append(b)
            index.setdefault(b, list()).append(a)

        return index

    @staticmethod
    def observation_to_space(observation: Observation) -> spaces.Space:
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        return spaces.Box(low, high, dtype=observation.dtype)


if __name__ == '__main__':
    env = PioneerEnv(headless=False, render_config=RenderConfig(camera_distance=4))
    env.reset_gui_camera()

    # env.reset_joint_positions(np.array([0.481, 1.033, -0.875, -0.850, -0.144, -0.689]))
    # print(env.scene.items_by_name['robot:pointer'].pose().xyz)
    #
    # env.reset_joint_positions(np.array([0.252, 1.036, -0.849, -1.083, -0.251, -0.742]))
    # print(env.scene.items_by_name['robot:pointer'].pose().xyz)

    # print(env.contacts())
    # print(env.unwanted_collisions_present())

    target_joint = env.scene.joints_by_name['robot:hinge1_to_arm1']
    env_velocity = 1.0
    target_joint.reset_state(target_joint.position(), velocity=env_velocity)

    # a = env.scene.items_by_name['robot:hinge1']
    # b = env.scene.items_by_name['robot:arm1']
    #
    # print(env.bullet.getDynamicsInfo(a.body_id, a.link_index))
    #
    # env.bullet.changeDynamics(bodyUniqueId=a.body_id, linkIndex=a.link_index, lateralFriction=0.0)
    # env.bullet.changeDynamics(bodyUniqueId=b.body_id, linkIndex=b.link_index, lateralFriction=0.0)

    count = 0
    while True:
        env.act(np.array([0, 1, 0, 0, 0, 0]), 0, 0)
        # target_joint.reset_state(target_joint.position(), velocity=env_velocity)
        #
        # rr = target_joint.upper_limit - target_joint.lower_limit
        # if target_joint.position() < target_joint.lower_limit + 0.01 * rr:
        #     env_velocity = 1.0
        #
        # if target_joint.position() > target_joint.upper_limit - 0.01 * rr:
        #     env_velocity = -1.0


        # for _ in range(env.world.frame_skip):
        #     print(f'{target_joint.position()} @ {target_joint.velocity()} - {target_joint.lower_limit}..{target_joint.upper_limit}')
        #     env.bullet.stepSimulation()

        sleep(env.world.step_time)
