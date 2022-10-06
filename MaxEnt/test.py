import gym
import numpy as np
from maxent import Env

q_table = np.load("q_table.npy")
env = Env(gym.make("MountainCar-v0", render_mode="human"), 20)

state = env.reset()
score = 0.
done = False
while not done:
    action = np.argmax(q_table[state[0]][state[1]])
    next_state, done = env.step(action)
    state = next_state
