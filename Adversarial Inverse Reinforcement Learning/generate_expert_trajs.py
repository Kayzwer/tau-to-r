import gymnasium as gym
import numpy as np
import readchar
import pickle


if __name__ == "__main__":
    arrow_keys = {
        "\x1b[D": 0,
        "\x1b[B": 1,
        "\x1b[C": 2
    }

    states = []
    actions = []
    next_states = []
    env = gym.make("MountainCar-v0", render_mode="human")
    for _ in range(10):
        state, _ = env.reset()
        done = False
        while not done:
            key = readchar.readkey()
            action = arrow_keys[key]
            next_state, _, done, _, _ = env.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            state = next_state

    with open("expert_trajs.pkl", "wb") as f:
        pickle.dump((np.array(states), np.array(
            actions).reshape(-1, 1), np.array(next_states)), f)
