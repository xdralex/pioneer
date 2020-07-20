from typing import Callable

import gym


class LambdaEnv(gym.Wrapper):
    def __init__(self, generator: Callable[[], gym.Env], env):
        super(LambdaEnv, self).__init__(env)

        self.generator = generator
        self.env = generator()

    def reset(self, **kwargs):
        self.env = self.generator()
