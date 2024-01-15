import torch
import numpy as np
from typing import Tuple
from torch.optim import Adam
from models import ActorNetwork
from torch.distributions import Categorical


class Agent:
    def __init__(self, state_dim: int, action_dim: int, gamma: float, lr: float
                 ) -> None:
        assert state_dim > 0
        assert action_dim > 0
        assert 0. < gamma < 1.
        assert 0. < lr < 1.
        self.network = ActorNetwork(state_dim, action_dim)
        self.optimizer = Adam(self.network.parameters(), lr)
        self.gamma = gamma
        self.trajs = []
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def choose_action(self, state: np.ndarray) -> Tuple[int, float]:
        action_prob = self.network(torch.from_numpy(state))
        action_dist = Categorical(action_prob)
        action = int(action_dist.sample().item())
        return action, float(action_dist.probs[action])

    def store_transition(self, state: np.ndarray, action: int, reward: float
                         ) -> None:
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def clear_memory(self) -> None:
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()

    def update(self) -> None:
        n = len(self.reward_memory)
        returns = torch.empty(n, dtype=torch.float32)
        returns_sum = 0.
        for i in range(n - 1, -1, -1):
            returns_sum = self.reward_memory[i] + self.gamma * returns_sum
            returns[i] = returns_sum
        returns = (returns - returns.mean()) / returns.std()
        self.optimizer.zero_grad()
        loss = torch.tensor(0.0, dtype=torch.float32)
        for state, action, return_ in zip(self.state_memory, self.action_memory, returns):
            loss -= torch.log(self.network(torch.from_numpy(state))
                              [action]) * return_
        loss.backward()
        self.optimizer.step()
        self.clear_memory()
