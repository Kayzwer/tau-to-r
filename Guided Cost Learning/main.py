from collect_trajectories import sample_trajectories, segregate_data
from torch.distributions import Categorical
from RewardFunction import RewardFunction
from REINFORCE import Agent
from torch import optim
import numpy as np
import torch
import gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, 0.0001, 0.99, 0.1)
    reward_function = RewardFunction(state_size + action_size - 1)
    reward_optimizer = optim.Adam(reward_function.parameters(), 0.0001,
                                  weight_decay=1e-4)
    demonstrations = np.load("demonstrations.npy", allow_pickle=True)
    log_cache = torch.tensor(1 / np.log(action_size), dtype=torch.float32)

    for demonstration in demonstrations:
        demonstration[:, -1] = 1.

    for i in range(5000000):
        traj = sample_trajectories(1, env, agent).squeeze(0).astype(np.float32)

        selected_demonstration_index = np.random.choice(len(demonstrations), 1)
        selected_demonstration = demonstrations[selected_demonstration_index][
            0].astype(np.float32)
        sample = np.vstack((traj, selected_demonstration))

        demo_states, demo_actions, _ = segregate_data(
            demonstration, state_size, action_size - 1)
        demo_rewards = reward_function(torch.cat((demo_states, demo_actions),
                                                 dim=1))

        sample_states, sample_actions, sample_action_probs = segregate_data(
            sample, state_size, action_size - 1)
        sample_rewards = reward_function(torch.cat((sample_states,
                                                    sample_actions), dim=1))

        reward_optimizer.zero_grad()
        reward_loss = torch.mean(demo_rewards) + torch.log(torch.mean(
            torch.exp(-sample_rewards) / (sample_action_probs + 1e-7)))
        reward_loss.backward()
        reward_optimizer.step()

        if i % 5 == 0:
            states, actions, _ = segregate_data(traj, state_size,
                                                action_size - 1)
            rewards = reward_function(torch.cat((states, actions), dim=1)
                                      ).detach().view(-1)
            returns = torch.zeros_like(rewards, dtype=torch.float32)
            returns_sum = 0.
            for j in range(len(returns) - 1, -1, -1):
                returns_sum = rewards[j] + agent.gamma * returns_sum
                returns[j] = returns_sum
            returns = (returns - returns.mean()) / returns.std()
            action_prob_dists = Categorical(agent.policy_network(states))
            log_probs = (action_prob_dists.log_prob(actions.view(-1)) *
                         log_cache)
            entropies = (action_prob_dists.entropy() * log_cache)
            loss = -(torch.sum(log_probs * returns) + torch.sum(entropies) *
                     agent.entropy_coef)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
        print(f"Iteration: {i + 1}, Traj_Step: {len(traj)}")
    torch.save(agent.policy_network.state_dict(), "IRL_Policy_Gradient.pt")
    torch.save(reward_function.state_dict(), "CartPole_Reward_Parameters.pt")
    env.close()
