import gym
import readchar
import numpy as np


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="human")
    arrow_keys = {
        '\x1b[D': 0,
        '\x1b[B': 1,
        '\x1b[C': 2,
    }

    trajectories = []
    for episode in range(20):
        trajectory = []
        state, _ = env.reset()
        done = False
        while not done:
            key = readchar.readkey()
            action = arrow_keys.get(key)
            trajectory.append((state[0], state[1], action))
            state, _, done, _, _ = env.step(action)
        trajectories.append(np.array(trajectory, dtype=np.float32))
    np.save("demonstrations", np.array(trajectories, dtype=np.object0))
