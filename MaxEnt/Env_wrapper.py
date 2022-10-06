import gym
import numpy as np
from typing import Tuple


class Env:
    def __init__(self, env: gym.Env, n_state_per_feature: int) -> None:
        self.env = env
        self.n_state_per_feature = n_state_per_feature
        env_high = env.observation_space.high
        self.env_low = env.observation_space.low
        self.env_unit = (env_high - self.env_low) / self.n_state_per_feature

    def get_state_idx(self, state: np.ndarray) -> Tuple[int, int]:
        position_idx = int((state[0] - self.env_low[0]) / self.env_unit[0])
        velocity_idx = int((state[1] - self.env_low[1]) / self.env_unit[1])
        return position_idx, velocity_idx

    def reset(self) -> Tuple[int, int]:
        return self.get_state_idx(self.env.reset()[0])

    def step(self, action: int) -> Tuple[Tuple[int, int], bool]:
        next_state, _, done, _, _ = self.env.step(action)
        return self.get_state_idx(next_state), done
