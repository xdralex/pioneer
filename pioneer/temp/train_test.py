import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["monitor"] = True
config["framework"] = 'torch'
config["num_gpus"] = 0
config["num_workers"] = 1
config["_fake_gpus"] = True
trainer = ppo.PPOTrainer(config=config, env="Reacher-v2")

for i in range(1000):
    result = trainer.train()
    print(pretty_print(result))
