from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pybullet
from pybullet_utils.bullet_client import BulletClient

from pioneer.collections_util import set_optional_kv
from pioneer.envs.bullet.bullet_bindings import BasePositionAndOrientation, LinkState, BaseVelocity, JointState


class Pose:
    def __init__(self,
                 bullet: BulletClient,
                 position: Tuple[float, float, float],
                 orientation: Tuple[float, float, float, float]):
        self.bullet = bullet

        self.position = position
        self.orientation = orientation

    def __repr__(self) -> str:
        return f'Pose(position={self.position}, orientation={self.orientation})'

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return self.position

    @property
    def rpy(self) -> Tuple[float, float, float]:
        return self.bullet.getEulerFromQuaternion(self.orientation)


@dataclass
class Velocity:
    linear: Tuple[float, float, float]
    angular: Tuple[float, float, float]


class Item(object):
    def __init__(self,
                 bullet: BulletClient,
                 name: Optional[str],
                 body_id: int,
                 link_index: Optional[int]):
        self.bullet = bullet
        self.name = name
        self.body_id = body_id
        self.link_index = link_index

    def __repr__(self) -> str:
        return f'Item(name={self.name}, body_id={self.body_id}, link_index={self.link_index})'

    def pose(self) -> Pose:
        if self.link_index is None:
            data = BasePositionAndOrientation(*self.bullet.getBasePositionAndOrientation(self.body_id))
            return Pose(self.bullet, data.position, data.orientation)
        else:
            data = LinkState(*self.bullet.getLinkState(self.body_id, self.link_index, computeLinkVelocity=1, computeForwardKinematics=1))
            return Pose(self.bullet, data.link_world_position, data.link_world_orientation)

    def velocity(self):
        if self.link_index is None:
            data = BaseVelocity(*self.bullet.getBaseVelocity(self.body_id))
            return Velocity(data.linear_velocity, data.angular_velocity)
        else:
            data = LinkState(*self.bullet.getLinkState(self.body_id, self.link_index, computeLinkVelocity=1, computeForwardKinematics=1))
            return Velocity(data.world_link_linear_velocity, data.world_link_angular_velocity)

    def reset_pose(self, position: Tuple[float, float, float], orientation: Tuple[float, float, float, float]):
        if self.link_index is not None:
            raise AssertionError('Position can be reset only for base items: to control linked items use joint methods')

        self.bullet.resetBasePositionAndOrientation(bodyUniqueId=self.body_id, posObj=position, ornObj=orientation)


class Joint(object):
    def __init__(self,
                 bullet: BulletClient,
                 item: Item,
                 name: Optional[str],
                 body_id: int,
                 joint_index: int,

                 joint_type: int,

                 damping: float,
                 friction: float,
                 max_force: float,

                 lower_limit: float,
                 upper_limit: float,
                 max_velocity: float):
        self.bullet = bullet
        self.item = item
        self.name = name
        self.body_id = body_id
        self.joint_index = joint_index

        self.joint_type = joint_type

        self.damping = damping
        self.friction = friction
        self.max_force = max_force

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.max_velocity = max_velocity

    def __repr__(self) -> str:
        return f'Joint(name={self.name}, body_id={self.body_id}, joint_index={self.joint_index}, ' \
               f'joint_type={self.joint_type}, ' \
               f'damping={self.damping}, friction={self.friction}, max_force={self.max_force}, ' \
               f'lower_limit={self.lower_limit}, upper_limit={self.upper_limit}, max_velocity={self.max_velocity})'

    def position(self) -> float:
        data = JointState(*self.bullet.getJointState(self.body_id, self.joint_index))
        return data.joint_position

    def velocity(self) -> float:
        data = JointState(*self.bullet.getJointState(self.body_id, self.joint_index))
        return data.joint_velocity

    def control_position(self,
                         position: float,
                         velocity: Optional[float] = None,
                         max_velocity: Optional[float] = None,
                         max_force: Optional[float] = None,
                         position_gain: Optional[float] = None,
                         velocity_gain: Optional[float] = None):
        kwargs = {
            'bodyUniqueId': self.body_id,
            'jointIndex': self.joint_index,
            'controlMode': pybullet.POSITION_CONTROL,
            'targetPosition': position
        }
        set_optional_kv(kwargs, 'targetVelocity', velocity)
        set_optional_kv(kwargs, 'maxVelocity', max_velocity)
        set_optional_kv(kwargs, 'force', max_force)
        set_optional_kv(kwargs, 'positionGain', position_gain)
        set_optional_kv(kwargs, 'velocityGain', velocity_gain)

        self.bullet.setJointMotorControl2(**kwargs)

    def control_velocity(self,
                         velocity: float,
                         max_force: Optional[float] = None):
        kwargs = {
            'bodyUniqueId': self.body_id,
            'jointIndex': self.joint_index,
            'controlMode': pybullet.VELOCITY_CONTROL,
            'targetVelocity': velocity
        }
        set_optional_kv(kwargs, 'force', max_force)

        self.bullet.setJointMotorControl2(**kwargs)

    def reset_state(self, position: float, velocity: Optional[float] = None):
        kwargs = {
            'bodyUniqueId': self.body_id,
            'jointIndex': self.joint_index,
            'targetValue': position
        }
        set_optional_kv(kwargs, 'targetVelocity', velocity)

        self.bullet.resetJointState(**kwargs)


class Scene:
    def __init__(self, bullet: BulletClient):
        self.bullet = bullet

        self.items: List[Item] = []
        self.items_by_name: Dict[str, Item] = {}

        self.joints: List[Joint] = []
        self.joints_by_name: Dict[str, Joint] = {}

    def add_item(self, item: Item):
        self.items.append(item)

        if item.name is not None:
            assert item.name not in self.items_by_name
            self.items_by_name[item.name] = item

    def add_joint(self, joint: Joint):
        self.joints.append(joint)

        if joint.name is not None:
            assert joint.name not in self.joints_by_name
            self.joints_by_name[joint.name] = joint

    def create_body_sphere(self,
                           name: Optional[str],
                           collision: bool,
                           mass: float,
                           radius: float,
                           position: Tuple[float, float, float],
                           orientation: Tuple[float, float, float, float],
                           rgba_color: Tuple[float, float, float, float]):

        visual_id = self.bullet.createVisualShape(self.bullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color)
        collision_id = self.bullet.createCollisionShape(self.bullet.GEOM_SPHERE, radius=radius) if collision else None

        kwargs = {
            'baseMass': mass,
            'baseVisualShapeIndex': visual_id,
            'basePosition': position,
            'baseOrientation': orientation
        }
        set_optional_kv(kwargs, 'baseCollisionShapeIndex', collision_id)

        body_id = self.bullet.createMultiBody(**kwargs)
        self.add_item(Item(bullet=self.bullet, name=name, body_id=body_id, link_index=None))

    def rpy2quat(self, rpy: Tuple[float, float, float]):
        return self.bullet.getQuaternionFromEuler(rpy)

    def quat2rpy(self, quat: Tuple[float, float, float, float]):
        return self.bullet.getEulerFromQuaternion(quat)

    def __repr__(self) -> str:
        items_str = '\n'.join([f'\t\t{x}' for x in self.items])
        joints_str = '\n'.join([f'\t\t{x}' for x in self.joints])

        return f'Scene(\n\titems: \n{items_str} \n\tjoints: \n{joints_str} \n)'


class World:
    def __init__(self, bullet: BulletClient, timestep: float, frame_skip: int, gravity: float):
        self.bullet = bullet

        self.timestep = timestep
        self.frame_skip = frame_skip
        self.gravity_force = gravity

        self.bullet.setGravity(0, 0, -self.gravity_force)
        self.bullet.setTimeStep(timestep)

    def step(self):
        for _ in range(self.frame_skip):
            self.bullet.stepSimulation()

    @property
    def step_time(self) -> float:
        return self.timestep * self.frame_skip
