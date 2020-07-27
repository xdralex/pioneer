import ray
from gym.wrappers import TimeLimit
from ray import tune
from ray.tune.registry import register_env
import pandas as pd

from pioneer.envs.pioneer import PioneerEnv


def train(results_dir: str,
          checkpoint_freq: int,
          num_samples: int,
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
                           'num_workers': 4,
                           'log_level': 'INFO',
                           'monitor': monitor,
                           'lr': tune.uniform(1e-5, 1e-4)
                       },
                       stop={
                           "training_iteration": 5
                       },
                       local_dir=results_dir,
                       checkpoint_freq=checkpoint_freq,
                       checkpoint_at_end=True)

    ray.shutdown()
    return results.dataframe()
