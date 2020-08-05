from typing import Dict, Any

import numpy as np
import pandas as pd
import ray
from gym.wrappers import TimeLimit
from ray import tune
from ray.tune.registry import register_env

from pioneer.envs.pioneer import PioneerConfig
from pioneer.envs.pioneer import PioneerEnv


def train_agent(results_dir: str,
                checkpoint_freq: int,
                num_samples: int,
                num_workers: int,
                monitor: bool) -> pd.DataFrame:

    def prepare_env(env_config: Dict[str, Any] = None):
        pioneer_config = PioneerConfig(
            award_done=env_config['award_done'],
            award_slope=env_config['award_slope'],
            penalty_step=env_config['penalty_step']
        )
        pioneer_env = PioneerEnv(pioneer_config=pioneer_config)
        return TimeLimit(pioneer_env, max_episode_steps=500)

    register_env('Pioneer-v1', prepare_env)
    ray.init(webui_host='0.0.0.0')

    def entropy_coeff_schedule(min_start_entropy: float,
                               max_start_entropy: float,
                               decay_steps: int,
                               base: float = 10):

        logmin = np.log(min_start_entropy) / np.log(base)
        logmax = np.log(max_start_entropy) / np.log(base)

        x = base ** (np.random.uniform(logmin, logmax))
        return [(0, x), (decay_steps, 0)]

    results = tune.run('PPO',
                       num_samples=num_samples,
                       config={
                           'env': 'Pioneer-v1',
                           'framework': 'torch',
                           'num_gpus': 0,
                           'num_workers': num_workers,
                           'log_level': 'INFO',
                           'monitor': monitor,

                           'env_config': {
                               'award_done': 5,
                               'award_slope': tune.loguniform(1, 10),
                               'penalty_step': 0.01
                           },

                           'model': {
                               'fcnet_hiddens': [256, 256]
                           },
                           'train_batch_size': 8000,
                           # 'entropy_coeff_schedule': [(0, 5e-2), (1000000, 0)],
                           # 'entropy_coeff_schedule': tune.sample_from(lambda _: entropy_coeff_schedule(1e-3, 1e-1, 1000000)),
                           'lr': tune.loguniform(1e-5, 1e-4),
                           'num_sgd_iter': 20,
                           'observation_filter': 'ConcurrentMeanStdFilter'
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
    train_agent(results_dir='~/Data/ray_results',
                checkpoint_freq=10,
                num_samples=1,
                num_workers=1,
                monitor=True)
