import torch
import pickle
import numpy as np
import gymnasium as gym
from REINFORCE import Agent
from models import AdversarialDiscriminator
from torch.nn.functional import binary_cross_entropy


if __name__ == "__main__":
    with open("./expert_trajs.pkl", "rb") as f:
        expert_states, expert_actions, expert_next_states = pickle.load(f)
        expert_states = torch.from_numpy(expert_states)
        expert_actions = torch.from_numpy(expert_actions)
        expert_next_states = torch.from_numpy(expert_next_states)
        expert_actions_probs = torch.ones((len(expert_actions), 1))
    env = gym.make("MountainCar-v0")
    gamma = .99
    agent = Agent(2, 3, gamma, .01)
    discriminator = AdversarialDiscriminator(2, 3, gamma, .001)
    iteration = 30000

    for i in range(iteration):
        agent_states = []
        agent_actions = []
        agent_next_states = []
        agent_actions_probs = []

        for _ in range(10):
            state, _ = env.reset()
            done = False
            while not done:
                action, action_prob = agent.choose_action(state)
                agent_actions.append(action)
                agent_actions_probs.append(action_prob)
                next_state, _, _, done, _ = env.step(action)
                agent_states.append(state)
                agent_next_states.append(next_state)
                state = next_state

        agent_states = torch.from_numpy(np.array(agent_states, dtype=np.float32)
                                        )
        agent_actions = torch.from_numpy(np.array(agent_actions, dtype=np.int64
                                                  ).reshape(-1, 1))
        agent_next_states = torch.from_numpy(np.array(agent_next_states,
                                                      dtype=np.float32))
        agent_actions_probs = torch.from_numpy(np.array(
            agent_actions_probs, dtype=np.float32).reshape(-1, 1))
        target = torch.concat(
            (torch.ones(len(expert_states), 1, dtype=torch.float32),
             torch.zeros(len(agent_states), 1, dtype=torch.float32)), dim=0)
        states = torch.concat((expert_states, agent_states), dim=0)
        actions = torch.concat((expert_actions, agent_actions), dim=0)
        next_states = torch.concat((expert_next_states, agent_next_states),
                                   dim=0)
        actions_probs = torch.concat((expert_actions_probs, agent_actions_probs
                                      ), dim=0)
        for _ in range(50):
            discriminator.optimizer.zero_grad()
            loss = binary_cross_entropy(discriminator(
                states, actions, next_states, actions_probs), target)
            loss.backward()
            discriminator.optimizer.step()
        state, _ = env.reset()
        done = False
        score = 0.
        while not done:
            action, action_prob = agent.choose_action(state)
            next_state, actual_reward, _, done, _ = env.step(action)
            reward = discriminator.get_reward(
                torch.from_numpy(state).unsqueeze(0), torch.tensor(
                    action, dtype=torch.int64).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(next_state).unsqueeze(0), torch.tensor(
                    action_prob).unsqueeze(0).unsqueeze(0)).item()
            agent.store_transition(state, action, reward)
            score += reward
            state = next_state
        agent.update()
        print(f"Episode: {i + 1}, Score: {score}")
