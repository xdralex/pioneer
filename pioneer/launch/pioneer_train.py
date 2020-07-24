import ray
import ray.rllib.agents.ppo as ppo
from gym.wrappers import TimeLimit
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from pioneer.envs.pioneer import PioneerEnv


def prepare_env():
    pioneer_env = PioneerEnv()
    return TimeLimit(pioneer_env, max_episode_steps=250)


register_env('Pioneer-v1', lambda _: prepare_env())

ray.init(webui_host='127.0.0.1')

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
