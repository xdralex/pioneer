import mujoco_py
import numpy as np
from gym import spaces

model = mujoco_py.load_model_from_path('pioneer/envs/assets/pioneer2.xml')
sim = mujoco_py.MjSim(model)

print(f'timestep: {model.opt.timestep}')

bounds = model.jnt_range.copy().astype(np.float32)
low, high = bounds.T
position_space = spaces.Box(low=low, high=high, dtype=np.float32)
print(f'bounds: {bounds}')

print(f'nq={model.nq}, nv={model.nv}')

a0 = sim.get_state()
print(f'qpos={a0.qpos}, nv={a0.qvel}')

a1 = mujoco_py.MjSimState(a0.time, a0.qpos, [0.2, -0.2], a0.act, a0.udd_state)
sim.set_state(a1)

sim.step()
sim.forward()

print(sim.data.qpos.flat[:])
print(sim.data.qvel.flat[:2])

exit(0)

#
# print(position_space.sample())
#
# sim.step()
#
# print(f"{sim.data.get_body_xpos('pointer')}")
#
# a0 = sim.get_state()
# print(a0)
#
# a1 = mujoco_py.MjSimState(a0.time, -1.0, 0.0, a0.act, a0.udd_state)
# print(a1)
# sim.set_state(a1)
#
# bounds = model.actuator_ctrlrange.copy().astype(np.float32)
# print(bounds)
# print(sim.data.ctrl)
#
# # sim.data.ctrl[:] = [10.0]
#
# sim.step()
# sim.forward()

# a1 = mujoco_py.MjSimState(a0.time, 0.0, 1.0, a0.act, a0.udd_state)
# sim.set_state(a1)
#
# sim.step()
# sim.forward()
#

viewer = mujoco_py.mjviewer.MjViewer(sim)

DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 0,
    'distance': 20.0,
    'lookat': np.array((0.0, 0.0, 0.0)),
    'elevation': -35.0,
    'azimuth': 135.0
}

for key, value in DEFAULT_CAMERA_CONFIG.items():
    if isinstance(value, np.ndarray):
        getattr(viewer.cam, key)[:] = value
    else:
        setattr(viewer.cam, key, value)

while True:
    sim.step()
    viewer.render()
    # print(f'{sim.get_state()} - {sim.data.get_body_xpos("pointer")}')
