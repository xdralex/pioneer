from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from pybullet_utils.bullet_client import BulletClient

from pioneer.robot.bullet_bindings import BasePositionAndOrientation, LinkState, BaseVelocity, JointState


class Pose:
    def __init__(self,
                 bullet: BulletClient,
                 position: Tuple[float, float, float],
                 orientation: Tuple[float, float, float, float]):
        self.bullet = bullet

        self.position = position
        self.orientation = orientation

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
                 name: str,
                 body_id: int,
                 link_index: Optional[int]):
        self.bullet = bullet
        self.name = name
        self.body_id = body_id
        self.link_index = link_index

    def world_pose(self) -> Pose:
        if self.link_index is None:
            data = BasePositionAndOrientation(*self.bullet.getBasePositionAndOrientation(self.body_id))
            return Pose(self.bullet, data.position, data.orientation)
        else:
            data = LinkState(*self.bullet.getLinkState(self.body_id, self.link_index))
            return Pose(self.bullet, data.link_world_position, data.link_world_orientation)

    def velocity(self):
        if self.link_index is None:
            data = BaseVelocity(*self.bullet.getBaseVelocity(self.body_id))
            return Velocity(data.linear_velocity, data.angular_velocity)
        else:
            data = LinkState(*self.bullet.getLinkState(self.body_id, self.link_index))
            return Velocity(data.world_link_linear_velocity, data.world_link_angular_velocity)


class Joint(object):
    def __init__(self,
                 bullet: BulletClient,
                 item: Item,
                 name: str,
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

    def position(self) -> float:
        data = JointState(*self.bullet.getJointState(self.body_id, self.joint_index))
        return data.joint_position

    def velocity(self) -> float:
        data = JointState(*self.bullet.getJointState(self.body_id, self.joint_index))
        return data.joint_position


class Scene:
    def __init__(self):
        self.items: List[Item] = []
        self.items_by_name: Dict[str, Item] = {}

        self.joints: List[Joint] = []
        self.joints_by_name: Dict[str, Joint] = {}

    def add_item(self, item: Item):
        assert item.name not in self.items_by_name

        self.items.append(item)
        self.items_by_name[item.name] = item

    def add_joint(self, joint: Joint):
        assert joint.name not in self.joints_by_name

        self.joints.append(joint)
        self.joints_by_name[joint.name] = joint
