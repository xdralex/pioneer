from collections import namedtuple


# Some of the records described in the PyBullet Quickstart Guide (http://goo.gl/QwJnFX)


BodyInfo = namedtuple('BodyInfo', ['body_name',     # bytes
                                   'model_name'])   # bytes


JointInfo = namedtuple('JointInfo', ['joint_index',             # int
                                     'joint_name',              # bytes
                                     'joint_type',              # int
                                     'q_index',                 # int
                                     'u_index',                 # int
                                     'flags',                   # int
                                     'joint_damping',           # float
                                     'joint_friction',          # float
                                     'joint_lower_limit',       # float
                                     'joint_upper_limit',       # float
                                     'joint_max_force',         # float
                                     'joint_max_velocity',      # float
                                     'link_name',               # bytes
                                     'joint_axis',              # Tuple[float, float, float]
                                     'parent_frame_pos',        # Tuple[float, float, float]
                                     'parent_frame_orn',        # Tuple[float, float, float, float)
                                     'parent_index'])           # int


BasePositionAndOrientation = namedtuple('BasePositionAndOrientation', ['position',          # Tuple[float, float, float]
                                                                       'orientation'])      # Tuple[float, float, float, float]


BaseVelocity = namedtuple('BaseVelocity', ['linear_velocity',       # Tuple[float, float, float]
                                           'angular_velocity'])     # Tuple[float, float, float]


LinkState = namedtuple('LinkState', ['link_world_position',                 # Tuple[float, float, float]
                                     'link_world_orientation',              # Tuple[float, float, float, float]
                                     'local_inertial_frame_position',       # Tuple[float, float, float]
                                     'local_inertial_frame_orientation',    # Tuple[float, float, float, float]
                                     'world_link_frame_position',           # Tuple[float, float, float]
                                     'world_link_frame_orientation',        # Tuple[float, float, float, float]
                                     'world_link_linear_velocity',          # Tuple[float, float, float]
                                     'world_link_angular_velocity'])        # Tuple[float, float, float]


JointState = namedtuple('JointState', ['joint_position',                # float
                                       'joint_velocity',                # float
                                       'joint_reaction_forces',         # Tuple[float, float, float, float, float, float]
                                       'applied_joint_motor_torque'])   # float


ContactPoint = namedtuple('ContactPoint', ['contact_flag',              # int
                                           'body_id_a',                 # int
                                           'body_id_b',                 # int
                                           'link_index_a',              # int
                                           'link_index_b',              # int
                                           'position_on_a',             # Tuple[float, float, float]
                                           'position_on_b',             # Tuple[float, float, float]
                                           'contact_normal_on_b',       # Tuple[float, float, float]
                                           'contact_distance',          # float
                                           'normal_force',              # float
                                           'lateral_friction1',         # float
                                           'lateral_friction_dir1',     # Tuple[float, float, float]
                                           'lateral_friction2',         # float
                                           'lateral_friction_dir2'])    # Tuple[float, float, float]


CameraImage = namedtuple('CameraImage', ['width',                       # int
                                         'height',                      # int
                                         'rgb_pixels',                  # List[Tuple[float, float, float, float]]
                                         'depth_pixels',                # List[float]
                                         'segmentation_mask_buffer'])   # List[int]
