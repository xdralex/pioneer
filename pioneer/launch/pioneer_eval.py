import datetime
import os
import pathlib
from contextlib import closing

import numpy as np
import ray
import ray.rllib.agents.ppo as ppo
from gym import spaces
from gym.wrappers import TimeLimit
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from ray.tune import register_env

from pioneer.envs.pioneer_env import PioneerSceneRandomizer, RandomizedPioneerEnv


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

    pioneer_env = RandomizedPioneerEnv(randomizer, temp_dir='/Users/xdralex/pioneer/environments', retain_samples=True)

    return TimeLimit(pioneer_env, max_episode_steps=250)


register_env('Pioneer-v1', lambda _: prepare_env())

ray.init()

config = ppo.DEFAULT_CONFIG.copy()

config['monitor'] = False
config['framework'] = 'torch'
config['num_gpus'] = 0
config['num_workers'] = 1
config['_fake_gpus'] = True
config['log_level'] = 'INFO'
config['evaluation_config'] = {'explore': False}

agent = ppo.PPOTrainer(config=config, env='Pioneer-v1')
agent.restore('/Users/xdralex/ray_results/PPO_Pioneer-v1_2020-07-20_16-57-06t7ouhjap/checkpoint_601/checkpoint-601')

print(agent.get_policy())
print(agent.get_policy().model)

env = prepare_env()

video_dir = '/Users/xdralex/pioneer/eval'
pathlib.Path(video_dir).mkdir(parents=True, exist_ok=True)

for i in range(10):
    now = datetime.datetime.now()
    filename = f'run_{now:%Y-%m-%d %H:%M:%S}'

    with closing(VideoRecorder(env, base_path=os.path.join(video_dir, filename))) as recorder:
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            recorder.capture_frame()
            episode_reward += reward

        print(f'{recorder.path} => {episode_reward}')
