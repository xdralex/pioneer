from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar, Union, Generic, Optional

import gym
import numpy as np
import pybullet
from pybullet_utils import bullet_client
from pybullet_utils.bullet_client import BulletClient

from pioneer.envs.bullet.bullet_bindings import JointInfo, BodyInfo
from pioneer.envs.bullet.bullet_scene import Scene, Item, Joint, World

Action = TypeVar('Action')
Observation = TypeVar('Observation')


@dataclass
class RenderConfig:
    camera_target: Tuple[float, float, float] = (0, 0, 0)

    camera_distance: float = 100.0

    camera_yaw: float = 120.0
    camera_pitch: float = -30.0
    camera_roll: float = 0.0

    render_width: int = 1280
    render_height: int = 800

    projection_fov: float = 30
    projection_near: float = 0.1
    projection_far: float = 200.0


@dataclass
class SimulationConfig:
    timestep: float = 1 / 240
    frame_skip: int = 10

    gravity: float = 0

    self_collision: bool = True
    collision_parent: bool = True

    @property
    def model_load_flags(self) -> int:
        flags = 0

        if self.self_collision:
            flags |= pybullet.URDF_USE_SELF_COLLISION

            if self.collision_parent:
                flags |= pybullet.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
            else:
                flags |= pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

        return flags

    @property
    def frames_per_second(self) -> int:
        return int(np.round(1 / (self.timestep * self.frame_skip)))


class BulletEnv(gym.Env, Generic[Action, Observation], ABC):
    def __init__(self,
                 model_path: str,
                 headless: bool = True,
                 simulation_config: Optional[SimulationConfig] = None,
                 render_config: Optional[RenderConfig] = None):
        self.model_path = model_path
        self.headless = headless
        self.simulation_config = simulation_config or SimulationConfig()
        self.render_config = render_config or RenderConfig()

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': self.simulation_config.frames_per_second
        }

        self.bullet: Optional[BulletClient] = None
        self.world: Optional[World] = None
        self.scene: Optional[Scene] = None

        self.world_index = -1
        self.step_index = 0

        self.reset_simulator()

    def reset_simulator(self):
        self.bullet = bullet_client.BulletClient(connection_mode=pybullet.DIRECT if self.headless else pybullet.GUI)
        self.world = World(bullet=self.bullet,
                           timestep=self.simulation_config.timestep,
                           frame_skip=self.simulation_config.frame_skip,
                           gravity=self.simulation_config.gravity)
        self.scene = self.load_scene(self.bullet, self.model_path, self.simulation_config)

        self.world_index += 1
        self.step_index = 0

    @staticmethod
    def load_scene(bullet: BulletClient, model_path: str, simulation_config: SimulationConfig) -> Scene:
        scene = Scene(bullet)

        object_ids = [bullet.loadURDF(model_path, flags=simulation_config.model_load_flags)]

        for body_id in object_ids:
            body_info = BodyInfo(*bullet.getBodyInfo(body_id))
            body_item = Item(bullet,
                             name=body_info.body_name.decode("utf8"),
                             body_id=body_id,
                             link_index=None)

            scene.add_item(body_item)

            for joint_index in range(bullet.getNumJoints(body_id)):
                joint_info = JointInfo(*bullet.getJointInfo(body_item.body_id, joint_index))

                joint_item = Item(bullet,
                                  name=joint_info.link_name.decode("utf8"),
                                  body_id=body_id,
                                  link_index=joint_index)

                joint = Joint(bullet,
                              item=joint_item,
                              name=joint_info.joint_name.decode("utf8"),
                              body_id=body_id,
                              joint_index=joint_index,

                              joint_type=joint_info.joint_type,

                              damping=joint_info.joint_damping,
                              friction=joint_info.joint_friction,
                              max_force=joint_info.joint_max_force,

                              lower_limit=joint_info.joint_lower_limit,
                              upper_limit=joint_info.joint_upper_limit,
                              max_velocity=joint_info.joint_max_velocity)

                scene.add_item(joint_item)
                if joint.joint_type == pybullet.JOINT_REVOLUTE:
                    scene.add_joint(joint)
                elif joint.joint_type == pybullet.JOINT_FIXED:
                    pass
                else:
                    raise AssertionError(f'Only revolute and fixed joints are supported atm, got: {joint_info}')

        return scene

    def reset_gui_camera(self):
        self.bullet.resetDebugVisualizerCamera(cameraDistance=self.render_config.camera_distance,
                                               cameraYaw=self.render_config.camera_yaw,
                                               cameraPitch=self.render_config.camera_pitch,
                                               cameraTargetPosition=self.render_config.camera_target)

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode == 'human':
            return None
        elif mode == 'rgb_array':
            view_matrix = self.bullet.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=list(self.render_config.camera_target),
                distance=self.render_config.camera_distance,

                yaw=self.render_config.camera_yaw,
                pitch=self.render_config.camera_pitch,
                roll=self.render_config.camera_roll,

                upAxisIndex=2)

            proj_matrix = self.bullet.computeProjectionMatrixFOV(
                fov=self.render_config.projection_fov,
                aspect=self.render_config.render_width / self.render_config.render_height,
                nearVal=self.render_config.projection_near,
                farVal=self.render_config.projection_far)

            (_, _, rgb_pixels, _, _) = self.bullet.getCameraImage(
                width=self.render_config.render_width,
                height=self.render_config.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

            return np.array(rgb_pixels)[:, :, :3]
        else:
            raise AssertionError(f'Render mode "{mode}" is not supported')

    def reset(self) -> Observation:
        self.reset_simulator()
        self.reset_world()
        return self.observe()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        self.step_index += 1

        reward, done, info = self.act(action, self.world_index, self.step_index)
        observation = self.observe()
        return observation, reward, done, info

    @abstractmethod
    def reset_world(self):
        pass

    @abstractmethod
    def act(self, action: Action, world_index: int, step_index: int) -> Tuple[float, bool, Dict]:
        pass

    @abstractmethod
    def observe(self) -> Observation:
        pass
