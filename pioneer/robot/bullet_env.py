from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar, List, Union, Generic, Optional

import gym
import numpy as np
import pybullet
from pybullet_utils import bullet_client
from pybullet_utils.bullet_client import BulletClient

from pioneer.robot.bullet_bindings import JointInfo, BodyInfo
from pioneer.robot.bullet_scene import Scene, Item, Joint

Action = TypeVar('Action')
Observation = TypeVar('Observation')


@dataclass
class BulletRenderConfig:
    camera_target: Tuple[float, float, float] = (0, 0, 0)

    camera_distance: float = 10.0

    camera_yaw: float = 120.0
    camera_pitch: float = -30.0
    camera_roll: float = 0.0

    render_width: int = 1280
    render_height: int = 800

    projection_fov: float = 60
    projection_near: float = 0.1
    projection_far: float = 100.0


@dataclass
class SimulationConfig:
    self_collision: bool = False
    collision_parent: bool = True

    @property
    def collision_flags(self) -> int:
        flags = 0
        if self.self_collision:
            flags |= pybullet.URDF_USE_SELF_COLLISION

            if self.collision_parent:
                flags |= pybullet.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
            else:
                flags |= pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

        return flags


class BulletEnv(gym.Env, Generic[Action, Observation]):
    def __init__(self,
                 model_path: str,
                 headless: bool = True,
                 simulation_config: Optional[SimulationConfig] = None,
                 render_config: Optional[BulletRenderConfig] = None):

        self.bullet: Optional[BulletClient] = None
        self.scene: Optional[Scene] = None

        self.model_path = model_path
        self.headless = headless
        self.simulation_config = simulation_config or SimulationConfig()
        self.render_config = render_config or BulletRenderConfig()

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 25
        }

    def reset_env(self):
        connection_mode = pybullet.DIRECT if self.headless else pybullet.GUI
        self.bullet = bullet_client.BulletClient(connection_mode=connection_mode)
        self.scene = Scene()

        object_ids = self.bullet.loadMJCF(self.model_path, flags=self.simulation_config.collision_flags)

        for body_id in object_ids:
            body_info = BodyInfo(*self.bullet.getBodyInfo(body_id))
            body_item = Item(self.bullet,
                             name=body_info.body_name.decode("utf8"),
                             body_id=body_id,
                             link_index=None)

            self.scene.add_item(body_item)

            for joint_index in range(self.bullet.getNumJoints(body_id)):
                joint_info = JointInfo(*self.bullet.getJointInfo(body_item.body_id, joint_index))

                joint_item = Item(self.bullet,
                                  name=joint_info.link_name,
                                  body_id=body_id,
                                  link_index=joint_index)

                joint = Joint(self.bullet,
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

                self.scene.add_item(joint_item)
                if joint.joint_type == pybullet.JOINT_REVOLUTE:
                    self.scene.add_joint(joint)
                elif joint.joint_type == pybullet.JOINT_FIXED:
                    pass
                else:
                    raise AssertionError(f'Only revolute and fixed joints are supported now, got: {joint_info}')

        # print(self.bullet.getBodyInfo(object_ids[1]))
        # print(self.bullet.getNumJoints(object_ids[1]))
        # for i in range(13):
        #     print(self.bullet.getJointInfo(object_ids[1], i))

    def seed(self, seed=None) -> List[int]:
        pass

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        pass

    def reset(self) -> Observation:
        pass

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

    def close(self):
        pass


if __name__ == '__main__':
    env = BulletEnv(model_path='/Users/xdralex/Work/curiosity/pioneer/pioneer/robot/assets/pioneer6.xml', headless=False)
    env.reset_env()

    while True:
        env.render()
