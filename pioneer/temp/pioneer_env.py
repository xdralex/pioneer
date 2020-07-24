import io
import math
import xml.etree.ElementTree as ET
from contextlib import closing
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Dict

import mujoco_py
import numpy as np
from gym import Space, Wrapper
from gym import spaces
from gym import utils

from pioneer.envs.mujoco_kinematic_env import MujocoKinematicEnv
from pioneer.xml_util import find_unique

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 0,
    'distance': 25.0,
    'lookat': np.array((0.0, 0.0, 0.0)),
    'elevation': -30.0,
    'azimuth': 120.0
}


@dataclass
class PioneerScene:
    model: bytes
    obstacle_pos: np.ndarray
    obstacle_size: np.ndarray


class PioneerEnv(MujocoKinematicEnv, utils.EzPickle):
    def __init__(self, model_path, scene: PioneerScene, config: Dict):
        self.potential_scale = float(config['potential_scale'])
        self.step_penalty = float(config['step_penalty'])
        self.stop_distance = float(config['stop_distance'])

        self.scene = scene
        self.best_distance = math.inf

        utils.EzPickle.__init__(self)
        MujocoKinematicEnv.__init__(self, model_path, frame_skip=20)

        self.position_space = self._prepare_position_space()

    def potential(self, distance: float) -> float:
        return 1 / (distance + 1 / self.potential_scale)

    def step(self, action: np.ndarray):
        pointer_coords = self.get_body_com('robot:pointer')
        target_coords = self.get_body_com('target')
        diff_coords = pointer_coords - target_coords
        distance = np.linalg.norm(diff_coords)

        reward = 0

        if distance < self.best_distance:
            reward += self.potential(distance) - self.potential(self.best_distance)
            self.best_distance = distance

        reward -= self.step_penalty

        self.do_simulation(action, self.frame_skip)
        observation = self._get_observation()
        done = distance < self.stop_distance

        return observation, reward, done, dict()

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def reset_model(self):
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self.best_distance = math.inf

        return self._get_observation()

    def _reset_sim(self):
        qpos = self.position_space.sample()
        qvel = np.zeros(self.model.nv)
        self.set_state(qpos, qvel)
        self.sim.forward()

        for _ in range(10):
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False

        return True

    def _prepare_position_space(self) -> Space:
        bounds = self.model.jnt_range.copy().astype(np.float32)
        assert bounds.shape == (self.model.nq, 2)

        low, high = bounds.T
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_observation(self):
        pointer_coords = self.get_body_com('robot:pointer')
        target_coords = self.get_body_com('target')
        pointer_target_diff = target_coords - pointer_coords

        state = self.sim.get_state()
        qpos = state.qpos.flat[:]
        qvel = state.qvel.flat[:]

        bounds = self.model.jnt_range.copy().astype(np.float32)
        low, high = bounds.T
        pos_low = low.flat[:]
        pos_high = high.flat[:]

        pos_distance = (pos_high - pos_low).flat[:]
        vel_to_pos = qvel / pos_distance

        return np.concatenate([
            qpos,
            pos_low,
            pos_high,

            qpos - pos_low,
            pos_high - qpos,

            qvel,
            vel_to_pos,

            np.cos(qpos), np.sin(qpos),
            np.cos(pos_low), np.sin(pos_low),
            np.cos(pos_high), np.sin(pos_high),

            pointer_coords,
            target_coords,
            pointer_target_diff
        ])


class PioneerSceneRandomizer(object):
    def __init__(self, source,
                 target_space: spaces.Space,
                 obstacle_pos_space: spaces.Space,
                 obstacle_size_space: spaces.Space):
        self.tree = ET.parse(source)

        self.target_space = target_space
        assert self.target_space.shape == (3,)

        self.obstacle_pos_space = obstacle_pos_space
        assert self.obstacle_pos_space.shape == (2,)

        self.obstacle_size_space = obstacle_size_space
        assert self.obstacle_size_space.shape == (3,)

        self.root = self.tree.getroot()
        assert self.root.tag == 'mujoco'

        self.worldbody = find_unique(self.root, 'worldbody')
        self.obstacles = find_unique(self.worldbody, 'body', attr_name='name', attr_value='obstacles')
        self.target = find_unique(self.worldbody, 'body', attr_name='name', attr_value='target')

    def generate_sample(self) -> PioneerScene:
        for child in list(self.obstacles):
            self.obstacles.remove(child)

        obstacle_size = list(self.obstacle_size_space.sample())
        obstacle_pos = list(self.obstacle_pos_space.sample()) + [obstacle_size[2]]

        obstacle_attrib = {
            'size': ' '.join([f'{x:0.4f}' for x in obstacle_size]),
            'pos': ' '.join([f'{x:0.4f}' for x in obstacle_pos]),
            'rgba': '0.3 0.3 0.3 1',
            'type': 'box'
        }
        ET.SubElement(self.obstacles, 'geom', obstacle_attrib)

        self.target.set('pos', ' '.join([f'{x:0.4f}' for x in self.target_space.sample()]))

        with closing(io.BytesIO()) as output:
            self.tree.write(output)
            model = output.getvalue()

        return PioneerScene(model=model,
                            obstacle_pos=np.array(obstacle_pos, dtype=np.float32),
                            obstacle_size=np.array(obstacle_size, dtype=np.float32))


class RandomizedPioneerEnv(Wrapper):
    def __init__(self, config: Dict, randomizer: PioneerSceneRandomizer, temp_dir: str, retain_samples: bool = False):
        self.config = config
        self.randomizer = randomizer
        self.temp_dir = temp_dir
        self.retain_samples = retain_samples

        super(RandomizedPioneerEnv, self).__init__(self._new_env())

    def reset(self, **kwargs):
        self.env.close()
        self.env = self._new_env()
        return self.env.reset()

    def _new_env(self):
        scene = self.randomizer.generate_sample()

        with NamedTemporaryFile(dir=self.temp_dir, delete=not self.retain_samples, suffix='.xml') as f:
            f.write(scene.model)
            f.flush()

            return PioneerEnv(f.name, scene, self.config)
