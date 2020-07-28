import ray
from gym.wrappers import TimeLimit
from ray import tune
from ray.tune.registry import register_env
import pandas as pd

from pioneer.envs.pioneer import PioneerEnv


def train(results_dir: str,
          checkpoint_freq: int,
          num_samples: int,
          num_workers: int,
          monitor: bool) -> pd.DataFrame:

    def prepare_env():
        pioneer_env = PioneerEnv()
        return TimeLimit(pioneer_env, max_episode_steps=250)

    register_env('Pioneer-v1', lambda _: prepare_env())

    ray.init(webui_host='0.0.0.0')

    results = tune.run('PPO',
                       num_samples=num_samples,
                       config={
                           'env': 'Pioneer-v1',
                           'framework': 'torch',
                           'num_gpus': 0,
                           'num_workers': num_workers,
                           'log_level': 'INFO',
                           'monitor': monitor,
                           'model': {
                               'fcnet_hiddens': [256, 128, 256]
                           },
                           'lr': 1e-5  # tune.loguniform(1e-6, 1e-4)
                       },
                       stop={
                           "training_iteration": 1000
                       },
                       local_dir=results_dir,
                       checkpoint_freq=checkpoint_freq,
                       checkpoint_at_end=True)

    ray.shutdown()
    return results.dataframe()


if __name__ == '__main__':
    train(results_dir='~/ray_results',
          checkpoint_freq=10,
          num_samples=1,
          num_workers=1,
          monitor=True)
