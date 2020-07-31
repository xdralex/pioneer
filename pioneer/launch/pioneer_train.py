from typing import Any, Dict

import ray
from gym.wrappers import TimeLimit
from ray import tune
from ray.tune.registry import register_env
import pandas as pd
import numpy as np

from pioneer.envs.pioneer import PioneerConfig
from pioneer.envs.pioneer import PioneerEnv


def train(results_dir: str,
          checkpoint_freq: int,
          num_samples: int,
          num_workers: int,
          monitor: bool) -> pd.DataFrame:

    def prepare_env(env_config: Dict[str, Any]):
        pioneer_config = PioneerConfig(
            award_potential_slope=float(env_config['award_potential_slope']),
            award_done=float(env_config['award_done']),
            penalty_step=float(env_config['penalty_step']),
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
                               'award_potential_slope': 10.0,
                               'award_done': 5.0,
                               'penalty_step': 1 / 100
                           },

                           'model': {
                               'fcnet_hiddens': [256, 256]
                           },
                           'train_batch_size': 8000,
                           'entropy_coeff_schedule': [(0, 5e-2), (1000000, 0)],
                           # 'entropy_coeff_schedule': tune.sample_from(lambda _: entropy_coeff_schedule(1e-3, 1e-1, 1000000)),
                           'lr': 2e-5,
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
    train(results_dir='~/Data/ray_results',
          checkpoint_freq=10,
          num_samples=1,
          num_workers=4,
          monitor=True)
