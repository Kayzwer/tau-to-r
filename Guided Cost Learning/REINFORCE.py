from typing import Tuple
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import gym


class Policy_Network(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Policy_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        ).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Agent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        lr: float,
        gamma: float,
        entropy_coef: float
    ) -> None:
        self.policy_network = Policy_Network(input_size, output_size)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.entropy_memory = []
        self.log_prob_memory = []
        self.reward_memory = []
        self.cache = torch.as_tensor(1 / np.log(output_size),
                                     dtype=torch.float64)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr)

    def choose_action_train(self, state: np.ndarray) -> int:
        action_prob_dist = Categorical(
            self.policy_network.forward(
                torch.as_tensor(state, dtype=torch.float64)))
        action = action_prob_dist.sample()
        log_prob = action_prob_dist.log_prob(action) * self.cache
        self.log_prob_memory.append(log_prob)
        self.entropy_memory.append(action_prob_dist.entropy() * self.cache)
        return action.detach().item()

    def choose_action_test(self, state: np.ndarray) -> int:
        return self.policy_network.forward(
            torch.as_tensor(state, dtype=torch.float64)
        ).detach().argmax().item()

    def get_action_and_prob(self, state: np.ndarray) -> Tuple[int, float]:
        softmax = self.policy_network(torch.as_tensor(
            state, dtype=torch.float64)).detach()
        action_probs = Categorical(softmax)
        action = action_probs.sample().item()
        prob = softmax[action].item()
        return action, prob

    def store_reward(self, reward: float) -> None:
        self.reward_memory.append(reward)

    def update(self) -> float:
        self.optimizer.zero_grad()
        T = len(self.reward_memory)
        returns = torch.zeros(T, dtype=torch.float64)
        returns_sum = 0.0
        for i in range(T - 1, -1, -1):
            returns_sum = self.reward_memory[i] + self.gamma * returns_sum
            returns[i] = returns_sum
        returns = (returns - returns.mean()) / returns.std()
        loss = torch.as_tensor(0.0, dtype=torch.float64)
        for return_, log_prob, entropy in zip(returns, self.log_prob_memory,
                                              self.entropy_memory):
            loss -= (return_ * log_prob + self.entropy_coef * entropy)
        loss.backward()
        self.optimizer.step()

        self.log_prob_memory.clear()
        self.reward_memory.clear()
        self.entropy_memory.clear()

        return loss.item()

    def train(self, env: gym.Env, iteration: int) -> None:
        for i in range(iteration):
            state, _ = env.reset()
            score = 0.0
            done = False
            while not done:
                action = self.choose_action_train(state)
                state, reward, done, _, _ = env.step(action)
                self.store_reward(reward)
                score += reward
            loss = self.update()
            print(f"Iteration: {i + 1}, Score: {score}, Loss: {loss}")
            if score >= 1000:
                break


def main():
    env = gym.make("CartPole-v1")
    agent = Agent(env.observation_space.shape[0], env.action_space.n, 0.001,
                  0.99, 0.01)
    agent.train(env, 10000)
    torch.save(agent.policy_network.state_dict(), "Policy_Gradient.pt")


if __name__ == "__main__":
    main()
