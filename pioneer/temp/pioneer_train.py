import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from gym import spaces
from gym.wrappers import TimeLimit
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from pioneer.envs.pioneer_env import RandomizedPioneerEnv, PioneerSceneRandomizer


def prepare_env():
    # randomizer = PioneerSceneRandomizer(source='/Users/xdralex/Work/curiosity/pioneer2/pioneer/envs/assets/pioneer4.xml',
    #                                     target_space=spaces.Box(low=np.array([5.0, -3, 1], dtype=np.float32),
    #                                                             high=np.array([6.0, 3, 3], dtype=np.float32)),
    #                                     obstacle_pos_space=spaces.Box(low=np.array([3, -2], dtype=np.float32),
    #                                                                   high=np.array([5, 2], dtype=np.float32)),
    #                                     obstacle_size_space=spaces.Box(low=np.array([0.1, 0.1, 3], dtype=np.float32),
    #                                                                    high=np.array([0.1, 0.1, 5], dtype=np.float32)))

    randomizer = PioneerSceneRandomizer(source='/Users/xdralex/Work/curiosity/pioneer/pioneer/envs/assets/pioneer6.xml',
                                        target_space=spaces.Box(low=np.array([5.0, -3, 1], dtype=np.float32),
                                                                high=np.array([6.0, 3, 3], dtype=np.float32)),
                                        obstacle_pos_space=spaces.Box(low=np.array([3, -2], dtype=np.float32),
                                                                      high=np.array([5, 2], dtype=np.float32)),
                                        obstacle_size_space=spaces.Box(low=np.array([0.001, 0.001, 0.001], dtype=np.float32),
                                                                       high=np.array([0.001, 0.001, 0.001], dtype=np.float32)))

    pioneer_config = {
        'potential_scale': 5,
        'step_penalty': 1 / 125,
        'stop_distance': 0.05
    }

    pioneer_env = RandomizedPioneerEnv(pioneer_config, randomizer, temp_dir='/Users/xdralex/pioneer/environments', retain_samples=True)

    return TimeLimit(pioneer_env, max_episode_steps=250)


register_env('Pioneer-v1', lambda _: prepare_env())

ray.init()

config = ppo.DEFAULT_CONFIG.copy()

config["monitor"] = True
config["framework"] = 'torch'
config["num_gpus"] = 0
config["num_workers"] = 1
config["_fake_gpus"] = True
config["log_level"] = 'INFO'

trainer = ppo.PPOTrainer(config=config, env='Pioneer-v1')
# trainer.restore('/Users/xdralex/ray_results/PPO_Pioneer-v1_2020-07-17_23-41-58nqnxyhom-4DOF-RND/checkpoint_1751/checkpoint-1751')

for i in range(10000):
    result = trainer.train()

    if i % 10 == 0:
        checkpoint = trainer.save()
        print(f'checkpoint: {checkpoint}')

    print(pretty_print(result))
