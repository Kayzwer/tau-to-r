from REINFORCE import Agent
import torch
import gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent(
        env.observation_space.shape[0], env.action_space.n, 0.01, 0.99, 0.02
    )
    agent.policy_network.load_state_dict(torch.load("IRL_Policy_Gradient.pt"))

    state, _ = env.reset()
    done = False
    score = 0.
    while not done:
        action = agent.choose_action_test(state)
        state, reward, done, _, _ = env.step(action)
        score += reward
    print(score)
