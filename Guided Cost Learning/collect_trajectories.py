from REINFORCE import Agent
from typing import Tuple
import numpy as np
import torch
import gym


def sample_trajectories(n: int, env: gym.Env, agent: Agent) -> np.ndarray:
    demonstrations = []
    for _ in range(n):
        trajectories = []
        state, _ = env.reset()
        done = False
        while not done:
            action, action_prob = agent.get_action_and_prob(state)
            trajectories.append([*state, action, action_prob])
            state, _, done, _, _ = env.step(action)
        demonstrations.append(np.asarray(trajectories, dtype=np.float32))
    return np.asarray(demonstrations, dtype=np.object0)


def segregate_data(data: np.ndarray, state_size: int, action_size: int
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.as_tensor(data[:, :state_size], dtype=torch.float32), \
        torch.as_tensor(data[:, state_size:state_size + action_size],
                        dtype=torch.float32), \
        torch.as_tensor(data[:, state_size + action_size:])


def main():
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, 0.001, 0.99, 0.01)
    agent.policy_network.load_state_dict(torch.load("Policy_Gradient.pt"))
    demonstrations = sample_trajectories(200, env, agent)
    np.save("demonstrations.npy", demonstrations)
    print("Demonstrations saved")


if __name__ == "__main__":
    main()
