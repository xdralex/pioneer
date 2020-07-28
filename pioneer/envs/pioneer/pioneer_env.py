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
        self.reinit_state()

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

            # 'ptr': arr2str(pointer_coords),
            # 'target': arr2str(target_coords),
            # 'diff': arr2str(diff),
            'dist': f'{distance:.3f}',
            'pot': f'{self.potential:.3f}',

            'a': arr2str(self.a),
            'v': arr2str(self.v),
            'r': arr2str(self.r)
            # 'j_act': arr2str(action),
            # 'j_pos': arr2str(self.joint_positions()),
            # 'j_lo': arr2str(np.array([x.lower_limit for x in self.scene.joints])),
            # 'j_hi': arr2str(np.array([x.upper_limit for x in self.scene.joints]))
        }

        # print(f'#{world_index}/#{step_index} - {dict2str(info, delim="  ")}')

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

    def reinit_state(self, positions: Optional[np.ndarray] = None):
        if positions is None:
            positions = self.np_random.uniform(self.r_lo, self.r_hi)

        self.a = np.zeros(self.dof, dtype=np.float32)
        self.v = np.zeros(self.dof, dtype=np.float32)
        self.r = positions

        self.reset_joint_positions(self.r)

        self.potential = 0

        # pointer_coords = np.array(self.scene.items_by_name['robot:pointer'].pose().xyz)
        # target_coords = np.array(self.scene.items_by_name['target'].pose().xyz)
        #
        # diff = target_coords - pointer_coords
        # distance = np.linalg.norm(diff)
        #
        # self.potential = self.compute_potential(distance)

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

    env.reinit_state(np.array([3.104, 0.027, -0.347, -2.077, 0.541, 2.732]))

    env.act(np.array([-0.647, -0.072, 0.459, -0.205, 0.453, -0.182]), 0, 0)
    env.act(np.array([-2.345, -0.166, -0.592, -9.230, -0.108, -2.552]), 0, 0)
    env.act(np.array([-2.956, -1.024, -0.265, -1.729, -0.157, 3.243]), 0, 0)
    env.act(np.array([-2.548, -0.466, -0.462, -5.976, -0.118, -2.421]), 0, 0)
    env.act(np.array([-1.413, -0.623, -0.679, 0.010, -0.217, -4.411]), 0, 0)
    env.act(np.array([-2.201, -0.488, -0.100, -1.033, -0.168, 5.141]), 0, 0)
    env.act(np.array([-2.158, -0.801, -0.489, -3.554, -0.146, -0.279]), 0, 0)
    env.act(np.array([-2.298, -0.907, -0.739, -1.607, -0.146, 2.889]), 0, 0)
    env.act(np.array([-2.205, -0.852, -0.149, 1.386, -0.057, -2.213]), 0, 0)
    env.act(np.array([-2.144, -0.764, -0.411, -11.437, -0.086, 0.955]), 0, 0)
    env.act(np.array([-2.460, -0.357, -0.220, -4.078, -0.113, 0.028]), 0, 0)
    env.act(np.array([-2.237, -0.326, -0.776, -0.347, -0.150, -3.410]), 0, 0)
    env.act(np.array([-2.118, -1.127, 0.447, 0.016, -0.313, 2.958]), 0, 0)
    env.act(np.array([-1.980, -0.467, -0.352, 2.616, -0.165, 3.145]), 0, 0)
    env.act(np.array([-2.149, -0.716, -0.621, -1.775, -0.405, 5.082]), 0, 0)
    env.act(np.array([-2.768, -0.279, -0.329, -2.337, -0.231, 1.150]), 0, 0)
    env.act(np.array([-2.842, -0.731, -0.213, 4.771, -0.203, 4.284]), 0, 0)
    env.act(np.array([-2.817, -0.577, 0.017, -7.489, -0.255, -3.161]), 0, 0)
    env.act(np.array([-2.717, -0.521, 0.044, -13.625, -0.406, -6.558]), 0, 0)
    env.act(np.array([-2.180, -0.236, -0.778, 4.983, -0.384, -4.787]), 0, 0)
    env.act(np.array([-2.897, -0.820, -1.125, -0.988, -0.194, -10.353]), 0, 0)
    env.act(np.array([-2.163, -0.470, -0.447, -4.620, -0.258, -0.005]), 0, 0)
    env.act(np.array([-1.666, -0.212, 0.096, -0.572, -0.317, 1.230]), 0, 0)
    env.act(np.array([-3.608, -0.323, -0.311, -6.418, -0.326, -3.372]), 0, 0)
    env.act(np.array([-2.281, -0.976, -0.209, -16.163, -0.256, 2.653]), 0, 0)
    env.act(np.array([-2.671, -0.008, 0.990, -6.174, -0.041, -7.007]), 0, 0)
    env.act(np.array([-3.101, -0.503, 0.144, -0.679, -0.271, -4.308]), 0, 0)
    env.act(np.array([-3.456, -0.315, -0.590, 5.617, -0.147, -11.568]), 0, 0)
    env.act(np.array([-1.837, -0.664, 0.465, -6.560, -0.003, 2.220]), 0, 0)
    env.act(np.array([-2.522, -0.734, 0.012, -1.307, -0.407, -6.126]), 0, 0)
    env.act(np.array([-3.149, -0.535, 0.459, -3.229, -0.125, 7.844]), 0, 0)
    env.act(np.array([-1.177, -0.631, -0.309, 6.367, -0.100, -7.552]), 0, 0)
    env.act(np.array([-2.788, -1.099, -1.238, 3.023, 0.011, 17.109]), 0, 0)
    env.act(np.array([-3.479, -0.262, -1.032, -17.877, -0.047, 0.445]), 0, 0)
    env.act(np.array([-1.873, -0.184, -0.495, 1.328, 0.236, -9.640]), 0, 0)
    env.act(np.array([-3.628, -1.584, -1.030, -0.187, 0.046, -16.845]), 0, 0)
    env.act(np.array([-1.251, -0.894, -1.213, -9.087, -0.200, -3.305]), 0, 0)
    env.act(np.array([-0.975, -0.811, -1.447, -2.425, -0.344, 6.658]), 0, 0)
    env.act(np.array([-2.168, -0.624, -0.615, -13.821, -0.235, -1.056]), 0, 0)
    env.act(np.array([-2.490, -0.667, -1.547, -2.574, -0.450, 3.702]), 0, 0)
    env.act(np.array([-2.110, -0.883, 0.828, 11.491, -0.214, -3.207]), 0, 0)
    env.act(np.array([-2.782, -0.902, -1.229, -2.099, -0.280, -4.795]), 0, 0)
    env.act(np.array([-2.521, -0.253, -0.451, 0.566, -0.253, -3.472]), 0, 0)
    env.act(np.array([-2.631, -0.267, 0.403, -6.818, -0.201, -2.484]), 0, 0)
    env.act(np.array([-3.184, -0.708, 0.504, -1.193, -0.139, 0.004]), 0, 0)
    env.act(np.array([-2.636, -0.572, 0.154, 3.879, 0.053, 0.939]), 0, 0)
    env.act(np.array([-2.497, -0.653, -0.346, -1.475, -0.132, -2.851]), 0, 0)
    env.act(np.array([-3.037, -0.873, -0.616, -2.263, -0.142, -1.924]), 0, 0)
    env.act(np.array([-3.198, -0.006, 0.135, 11.225, 0.193, -2.371]), 0, 0)
    env.act(np.array([-3.358, -0.690, -0.840, 2.044, 0.052, -0.362]), 0, 0)
    env.act(np.array([-3.783, -0.620, 0.982, 6.040, 0.123, -7.191]), 0, 0)
    env.act(np.array([-3.210, -0.644, -0.376, -3.982, 0.596, -1.506]), 0, 0)
    env.act(np.array([-3.349, -1.757, -0.618, -2.501, -0.146, -5.245]), 0, 0)
    env.act(np.array([-2.998, -0.601, -0.335, -1.513, 0.464, -3.228]), 0, 0)
    env.act(np.array([-2.620, -0.885, -0.692, -2.093, 1.333, -1.732]), 0, 0)
    env.act(np.array([-1.827, -0.977, -0.227, -3.679, 0.773, -0.751]), 0, 0)
    env.act(np.array([0.034, -0.684, -2.451, -7.935, 1.719, -2.100]), 0, 0)
    env.act(np.array([-1.960, -1.998, -2.319, 0.301, -0.110, -1.762]), 0, 0)
    env.act(np.array([0.369, 0.874, -2.819, -5.313, -2.441, -1.303]), 0, 0)
    env.act(np.array([0.041, -2.598, -1.916, -1.698, 3.038, -1.547]), 0, 0)
    env.act(np.array([-0.710, -3.619, -2.470, -0.708, 1.500, -2.778]), 0, 0)
    env.act(np.array([-1.563, 0.146, 0.405, -3.155, 3.030, -1.013]), 0, 0)
    env.act(np.array([0.671, -1.209, -1.507, -0.050, 1.902, -2.084]), 0, 0)
    env.act(np.array([0.176, -0.911, -1.787, -2.306, 0.657, -0.142]), 0, 0)
    env.act(np.array([-0.836, -0.789, -0.904, 1.990, 2.179, -1.452]), 0, 0)
    env.act(np.array([-0.261, -0.424, -2.298, 0.292, 2.577, -1.204]), 0, 0)
    env.act(np.array([-0.369, -0.685, -1.120, -2.913, 2.329, -2.316]), 0, 0)
    env.act(np.array([-0.692, -3.341, -0.883, -1.714, 0.207, -1.011]), 0, 0)
    env.act(np.array([-0.968, -1.443, -3.657, -0.875, -0.087, -2.674]), 0, 0)
    env.act(np.array([-1.184, -2.585, -1.402, -2.910, 2.114, -1.954]), 0, 0)
    env.act(np.array([0.183, -1.757, -0.793, -2.176, 1.535, -3.160]), 0, 0)
    env.act(np.array([-0.266, -2.762, -1.022, 0.344, 1.513, -1.135]), 0, 0)
    env.act(np.array([-0.518, -0.590, -1.094, -0.331, 1.191, -2.386]), 0, 0)
    env.act(np.array([-1.189, 0.018, -1.468, 0.081, 0.733, -1.350]), 0, 0)
    env.act(np.array([-0.996, -0.687, -0.202, 3.448, 0.688, -1.521]), 0, 0)
    env.act(np.array([-0.779, -1.805, -1.090, 0.754, 1.819, -0.175]), 0, 0)
    env.act(np.array([-1.142, -1.029, -1.055, -2.610, -1.042, -2.413]), 0, 0)
    env.act(np.array([-0.963, -1.467, -0.761, 0.808, -1.295, 0.667]), 0, 0)
    env.act(np.array([-0.377, 0.392, -0.679, -3.643, -0.152, -0.780]), 0, 0)
    env.act(np.array([-2.416, -0.137, -2.249, -1.116, 1.019, -0.346]), 0, 0)
    env.act(np.array([0.253, -2.524, 0.025, -1.171, 1.295, 1.046]), 0, 0)
    env.act(np.array([0.199, -0.209, 0.613, -4.753, 0.312, 0.193]), 0, 0)
    env.act(np.array([-1.468, -0.989, -0.674, -1.727, 2.304, -0.760]), 0, 0)
    env.act(np.array([-0.988, -0.298, -0.845, -0.265, 0.946, 0.633]), 0, 0)
    env.act(np.array([-1.479, 0.143, 0.708, 3.838, -0.923, 0.849]), 0, 0)
    env.act(np.array([-0.145, -0.765, 0.114, -1.725, -0.059, -0.169]), 0, 0)
    env.act(np.array([-0.637, -2.353, -0.076, -0.780, 0.127, -0.316]), 0, 0)
    env.act(np.array([0.577, -0.159, 0.804, -0.884, 0.120, 0.576]), 0, 0)
    env.act(np.array([-2.067, 0.656, 0.439, -0.575, 0.347, 0.593]), 0, 0)
    env.act(np.array([-1.083, -1.156, -0.183, 1.517, 0.353, -3.520]), 0, 0)
    env.act(np.array([0.796, -1.091, -0.464, 1.113, -0.449, 0.583]), 0, 0)
    env.act(np.array([-0.081, 0.493, 0.294, -1.633, 0.410, 0.057]), 0, 0)
    env.act(np.array([0.450, -0.465, 0.415, -0.528, 0.647, -0.890]), 0, 0)
    env.act(np.array([-0.325, -0.653, -1.544, 2.040, -0.326, 0.152]), 0, 0)
    env.act(np.array([-1.135, -1.036, 0.314, -1.536, 0.250, -0.256]), 0, 0)
    env.act(np.array([0.008, -1.797, -0.065, -1.990, 0.161, 1.309]), 0, 0)
    env.act(np.array([1.643, -1.155, -0.436, 1.854, 0.404, 0.144]), 0, 0)
    env.act(np.array([-0.377, 0.271, 0.587, 0.389, -0.238, 0.949]), 0, 0)
    env.act(np.array([1.008, -1.160, 0.501, -1.336, 0.326, -0.489]), 0, 0)
    env.act(np.array([-0.706, 0.002, -1.140, 0.390, 0.048, -0.648]), 0, 0)
    env.act(np.array([-0.492, -0.201, 0.522, -0.950, -0.026, 0.403]), 0, 0)
    env.act(np.array([0.828, -0.956, 1.333, 1.138, 0.678, -0.975]), 0, 0)
    env.act(np.array([-1.470, -0.681, -0.299, -1.436, -0.155, -1.151]), 0, 0)
    env.act(np.array([-1.199, -1.183, -0.529, -1.312, 0.319, -1.075]), 0, 0)
    env.act(np.array([-2.292, -0.746, 1.047, -0.911, -0.620, 0.019]), 0, 0)
    env.act(np.array([0.108, 0.103, -0.085, -0.211, -0.153, 1.334]), 0, 0)
    env.act(np.array([0.569, 0.366, -0.225, -0.642, -0.265, -0.465]), 0, 0)
    env.act(np.array([-1.286, -0.057, -0.370, -2.503, -0.285, -0.412]), 0, 0)
    env.act(np.array([0.404, -0.919, 0.278, -0.395, 0.326, 0.012]), 0, 0)
    env.act(np.array([-1.025, -0.195, -0.826, -1.444, -0.174, 0.273]), 0, 0)
    env.act(np.array([0.274, -1.028, 0.940, 2.136, -0.299, 1.183]), 0, 0)
    env.act(np.array([-1.454, -0.903, -1.281, -0.256, 0.665, -1.086]), 0, 0)
    env.act(np.array([-0.398, 0.303, -0.009, 2.466, 0.229, 1.579]), 0, 0)
    env.act(np.array([-0.634, -0.317, -0.177, -0.708, -0.453, 0.006]), 0, 0)
    env.act(np.array([-0.890, -0.885, 0.024, -0.077, 0.188, 0.367]), 0, 0)
    env.act(np.array([-0.943, -1.442, -1.546, -1.338, 0.321, -0.345]), 0, 0)
    env.act(np.array([-1.176, -0.773, 0.405, -2.379, 0.156, -0.092]), 0, 0)
    env.act(np.array([-0.425, 0.015, 0.651, -0.570, 0.615, -0.021]), 0, 0)
    env.act(np.array([0.673, -1.626, -0.818, 0.178, 0.060, -0.851]), 0, 0)
    env.act(np.array([0.670, -0.232, 0.118, -0.636, 0.201, 0.609]), 0, 0)
    env.act(np.array([-2.010, 0.453, 0.561, -2.291, -0.583, -1.102]), 0, 0)
    env.act(np.array([-0.262, -0.700, -0.783, -0.317, 0.228, 0.142]), 0, 0)
    env.act(np.array([-0.769, 1.227, -0.055, 0.070, -0.057, 1.538]), 0, 0)
    env.act(np.array([-1.570, -1.036, -0.256, 1.446, 0.686, 0.143]), 0, 0)
    env.act(np.array([-0.154, -1.942, -0.684, -0.759, 0.316, -1.133]), 0, 0)
    env.act(np.array([-1.138, -1.951, 0.823, -1.932, 0.338, -0.263]), 0, 0)
    env.act(np.array([-0.573, -2.186, 0.077, -1.459, -0.212, -0.584]), 0, 0)
    env.act(np.array([1.004, -1.707, 0.660, -1.717, 0.037, -0.719]), 0, 0)
    env.act(np.array([-0.862, -1.148, 0.895, -0.444, 0.725, 0.045]), 0, 0)
    env.act(np.array([-0.514, -1.245, -0.634, -0.414, -0.457, -0.827]), 0, 0)
    env.act(np.array([-1.708, -0.259, -0.814, -0.161, 0.228, -2.516]), 0, 0)
    env.act(np.array([0.107, -1.326, -1.021, -0.045, 0.862, -0.945]), 0, 0)
    env.act(np.array([-1.880, 0.509, 1.452, -0.048, 0.482, -0.620]), 0, 0)
    env.act(np.array([-0.421, -0.819, 0.677, -2.462, 0.432, 0.393]), 0, 0)
    env.act(np.array([-0.522, -0.608, -0.386, -2.585, -0.357, 1.000]), 0, 0)
    env.act(np.array([-2.028, -1.760, -0.718, -2.530, 0.480, -1.200]), 0, 0)
    env.act(np.array([-1.587, -1.575, -0.099, 1.247, 0.233, -0.984]), 0, 0)
    env.act(np.array([-0.246, -0.257, -0.520, -0.757, -0.078, 0.275]), 0, 0)
    env.act(np.array([0.210, -0.147, -0.548, -0.627, 0.522, 0.179]), 0, 0)
    env.act(np.array([0.390, -0.989, 0.397, 0.229, 0.844, -0.530]), 0, 0)
    env.act(np.array([-2.243, -0.250, 0.568, -2.051, 0.349, 1.936]), 0, 0)
    env.act(np.array([-0.741, -0.457, 0.283, 1.682, -0.444, -0.924]), 0, 0)
    env.act(np.array([0.058, 0.161, -0.269, -0.520, -0.168, 0.504]), 0, 0)
    env.act(np.array([-1.169, 0.522, 0.135, 0.589, 0.735, 1.947]), 0, 0)
    env.act(np.array([-0.100, 0.873, 0.843, -0.016, 1.150, 0.721]), 0, 0)
    env.act(np.array([-1.823, -0.432, -0.320, -1.478, 0.262, -0.806]), 0, 0)
    env.act(np.array([0.851, -0.690, -0.333, 0.490, 0.122, -0.807]), 0, 0)
    env.act(np.array([1.004, -0.576, 0.488, -0.870, 0.075, 0.670]), 0, 0)
    env.act(np.array([-1.032, -1.316, 0.183, -0.015, 0.061, 0.034]), 0, 0)
    env.act(np.array([-0.134, -0.569, 0.736, -0.102, 0.374, 0.191]), 0, 0)
    env.act(np.array([0.202, 0.215, -0.212, 0.484, -0.733, 0.454]), 0, 0)
    env.act(np.array([1.252, -0.628, 0.569, 1.008, 0.485, -1.195]), 0, 0)
    env.act(np.array([0.219, -0.556, 0.422, 0.130, -0.049, 0.701]), 0, 0)
    env.act(np.array([-0.939, -0.061, -0.477, 1.480, 0.066, 0.307]), 0, 0)
    env.act(np.array([-0.014, -1.991, -0.528, -2.175, 0.370, 1.834]), 0, 0)
    env.act(np.array([-0.133, -1.148, 0.122, 0.962, -0.229, 1.982]), 0, 0)
    env.act(np.array([-1.779, -1.691, -0.696, -0.137, 0.417, -0.478]), 0, 0)
    env.act(np.array([-1.824, 0.215, 0.646, 0.046, 0.159, 0.171]), 0, 0)
    env.act(np.array([-0.948, -0.716, -0.107, -2.179, 0.773, 0.586]), 0, 0)
    env.act(np.array([-1.049, -1.262, -0.079, -0.816, 0.948, 0.192]), 0, 0)
    env.act(np.array([-1.615, 0.450, -0.745, 0.782, 0.449, 0.129]), 0, 0)
    env.act(np.array([-0.090, -0.948, 1.001, -0.443, -0.207, 0.826]), 0, 0)
    env.act(np.array([-1.289, -2.019, 0.107, 2.075, 0.679, 1.131]), 0, 0)
    env.act(np.array([0.954, -1.118, -0.394, 0.818, -0.257, -0.310]), 0, 0)

    # env.reinit_state(np.array([-3.142, -1.309, -1.309, -3.142, 1.571, -3.142]))
    # print(env.joint_positions())
    #
    # env.act(np.array([-1.023, -1.009, 0.489, 1.018, -0.121, -0.019]), 0, 0)
    # print(env.joint_positions())
    #
    # env.act(np.array([-0.596, -0.003, -0.533, 0.083, -0.129, 0.202]), 0, 0)
    # print(env.joint_positions())
    #
    # env.act(np.array([-1.936, 0.253, -0.804, 0.498, 0.819, 0.827]), 0, 0)
    # print(env.joint_positions())
    #
    # env.act(np.array([0.641, -0.664, -0.020, 0.762, 0.152, -2.680]), 0, 0)
    # print(env.joint_positions())


    # env.world.step()

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

    # while True:
    #     # env.act(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), world_index=0, step_index=0)
    #
    #     # rr = target_joint.upper_limit - target_joint.lower_limit
    #     # if target_joint.position() < target_joint.lower_limit + 0.01 * rr:
    #     #     target_joint.control_velocity(velocity=1.0)
    #     #
    #     # if target_joint.position() > target_joint.upper_limit - 0.01 * rr:
    #     #     target_joint.control_velocity(velocity=-1.0)
    #
    #     # target_joint.control_velocity(velocity=10.0, max_force=1000)
    #
    #     # target_joint.control_position(pos, position_gain=0.1, velocity_gain=0.1, max_force=0)
    #     # pos += diff / 96
    #
    #     # print(f'{pos} - {target_joint.position()}')
    #
    #     # print(f'{target_joint.position()}')
    #
    #     # env.act(np.array([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0]), world_index=0, step_index=0)
    #
    #     env.world.step()
    #     sleep(env.world.step_time)
    #
    #     env.act(np.array([-0.364, 0.459, 0.658, 0.506, 0.983, 0.373]), 0, 0)
    #     print(env.joint_positions())
