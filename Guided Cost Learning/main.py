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
    agent = Agent(state_size, action_size, 0.0001, 0.99, 0.02)
    reward_function = RewardFunction(state_size + action_size - 1)
    reward_optimizer = optim.Adam(reward_function.parameters(), 0.0001)
    demonstrations = np.load("demonstrations.npy", allow_pickle=True)
    samples = np.empty(0, dtype=np.object0)
    log_cache = torch.tensor(1 / np.log(action_size), dtype=torch.float32)

    DEMO_BATCH = 50
    TRAJ_TO_SAMPLE = 50

    for i in range(500):
        trajs = sample_trajectories(TRAJ_TO_SAMPLE, env, agent)
        samples = np.append(samples, trajs)
        average_step = 0.0
        for traj in trajs:
            average_step += len(traj)
        average_step /= TRAJ_TO_SAMPLE
        for _ in range(50):
            selected_samples_index = np.random.choice(len(samples), DEMO_BATCH)
            selected_demonstrations_index = np.random.choice(len(demonstrations
                                                                 ), DEMO_BATCH)

            selected_samples = samples[selected_samples_index]
            selected_demonstrations = demonstrations[
                selected_demonstrations_index]
            selected_samples = np.append(selected_samples,
                                         selected_demonstrations)

            total_demo_reward = torch.tensor(0.0, dtype=torch.float32)
            for demonstration in selected_demonstrations:
                states, actions, _ = segregate_data(demonstration, state_size,
                                                    action_size - 1)
                demo_reward = reward_function(torch.cat((states, actions),
                                                        dim=1))
                if i < 10:
                    demo_reward = (demo_reward - demo_reward.mean()) / \
                        demo_reward.std()
                total_demo_reward += torch.sum(demo_reward)

            total_partition = torch.tensor(0.0, dtype=torch.float32)
            total_partition_weight = torch.tensor(0.0, dtype=torch.float32)
            for sample in selected_samples:
                states, actions, action_probs = segregate_data(
                    sample, state_size, action_size - 1)
                sample_reward = reward_function(torch.cat((states, actions),
                                                          dim=1))
                if i < 10:
                    sample_reward = (sample_reward - sample_reward.mean()) / \
                        sample_reward.std()
                sample_reward = torch.sum(sample_reward)
                partition_weight = torch.exp(sample_reward) / \
                    torch.prod(action_probs).clamp_min(1e-5)
                total_partition_weight += partition_weight
                total_partition += partition_weight.detach() * sample_reward
            reward_optimizer.zero_grad()
            reward_loss = total_demo_reward / DEMO_BATCH - total_partition / \
                total_partition_weight
            reward_loss.backward()
            reward_optimizer.step()

        loss = torch.tensor(0.0, dtype=torch.float32)
        for traj in trajs:
            states, actions, _ = segregate_data(
                traj, state_size, action_size - 1)
            rewards = reward_function(
                torch.cat((states, actions), dim=1)).detach().view(-1)
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
            loss -= (torch.sum(log_probs * returns) + torch.sum(entropies)
                     * agent.entropy_coef)
        loss /= TRAJ_TO_SAMPLE
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        torch.save(agent.policy_network.state_dict(), "IRL_Policy_Gradient.pt")
        torch.save(reward_function.state_dict(),
                   "CartPole_Reward_Parameters.pt")
        print(f"Iteration: {i + 1} done, Average Step: {average_step}")
    env.close()
