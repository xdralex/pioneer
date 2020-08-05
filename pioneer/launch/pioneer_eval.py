import datetime
import os
import pathlib
from contextlib import closing
from typing import Dict, Any

import ray
import ray.rllib.agents.ppo as ppo
from gym.wrappers import TimeLimit
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from ray.rllib.agents.trainer import with_base_config
from ray.tune import register_env

from pioneer.envs.pioneer import PioneerConfig
from pioneer.envs.pioneer import PioneerEnv


def eval_agent(checkpoint_file: str):
    def prepare_env(env_config: Dict[str, Any]):
        pioneer_config = PioneerConfig(
            award_done=env_config['award_done'],
            award_slope=env_config['award_slope'],
            penalty_step=env_config['penalty_step']
        )
        pioneer_env = PioneerEnv(pioneer_config=pioneer_config)
        return TimeLimit(pioneer_env, max_episode_steps=500)

    register_env('Pioneer-v1', prepare_env)
    ray.init(webui_host='0.0.0.0')

    config = with_base_config(ppo.DEFAULT_CONFIG, {
        'env': 'Pioneer-v1',
        'framework': 'torch',
        'num_gpus': 0,
        'num_workers': 1,
        'log_level': 'INFO',
        'monitor': True,

        'evaluation_config': {'explore': False},

        'env_config': {
            'award_done': 5,
            'award_slope': 1,
            'penalty_step': 0.01
        },
    })

    agent = ppo.PPOTrainer(config=config, env='Pioneer-v1')
    agent.restore(checkpoint_file)

    env = prepare_env(config['env_config'])

    checkpoint_dir = pathlib.Path(checkpoint_file).parent
    video_dir = os.path.join(checkpoint_dir, 'eval')
    pathlib.Path(video_dir).mkdir(parents=True, exist_ok=True)

    for i in range(10):
        now = datetime.datetime.now()
        filename = f'run_{now:%Y-%m-%d %H:%M:%S}'

        with closing(VideoRecorder(env, base_path=os.path.join(video_dir, filename))) as recorder:
            episode_reward = 0
            done = False
            env.seed()
            obs = env.reset()
            while not done:
                action = agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
                recorder.capture_frame()
                episode_reward += reward

            print(f'{recorder.path} => {episode_reward}')


if __name__ == '__main__':
    eval_agent(checkpoint_file='/Users/xdralex/Data/checkpoints/0001 - velocity_no_obstacles/checkpoint-1000')
