from Env_wrapper import Env, gym, np, Tuple
from typing import List


def update_q_table(q_table: np.ndarray, gamma: float, alpha: float,
                   state: Tuple[int, int], action: int, reward: float,
                   next_state: Tuple[int, int]) -> None:
    q_1 = q_table[state[0]][state[1]][action]
    q_2 = reward + gamma * max(q_table[next_state[0]][next_state[1]])
    q_table[state[0]][state[1]][action] += alpha * (q_2 - q_1)


def get_reward(theta: np.ndarray, state: Tuple[int, int]) -> float:
    return theta[state[0]][state[1]]


def maxent_irl(expert: np.ndarray, learner: np.ndarray, theta: np.ndarray,
               lr: float, func: np.vectorize) -> None:
    gradient = expert - learner
    theta += lr * gradient
    theta = func(theta)


def ReLU(x: float) -> float:
    return max(0., x)


def expert_feature_expectations(demonstrations: np.ndarray) -> np.ndarray:
    feature_expectations = np.zeros((20, 20))
    for demonstration in demonstrations:
        for state_1, state_2, _ in demonstration:
            feature_expectations[int(state_1)][int(state_2)] += 1
    return feature_expectations / len(demonstrations)


def discretize_states(env: Env, demonstrations: List) -> np.ndarray:
    n = len(demonstrations)
    demos = np.zeros((n, get_max_len(demonstrations),
                      env.env.observation_space.shape[0] + 1))
    for i in range(n):
        for j in range(len(demonstrations[i])):
            demos[i][j][0], demos[i][j][1] = demonstrations[i][j][0:2]
            demos[i][j][2] = demonstrations[i][j][2]
    return demonstrations


def get_max_len(demonstrations: List) -> int:
    max_ = 0
    for demonstration in demonstrations:
        n = len(demonstration)
        if n > max_:
            max_ = n
    return max_


if __name__ == "__main__":
    n_state_per_feature = 20
    env = Env(gym.make("MountainCar-v0"), n_state_per_feature)
    n_actions = env.env.action_space.n
    theta = -np.random.uniform(size=(n_state_per_feature, n_state_per_feature))
    q_table = np.zeros((n_state_per_feature, n_state_per_feature, n_actions))
    gamma = .99
    alpha = 3e-2
    lr = 5e-2

    demonstrations = discretize_states(env, np.load("demonstrations.npy",
                                                    allow_pickle=True))
    expert = expert_feature_expectations(demonstrations)
    learner_feature_expectations = np.zeros((n_state_per_feature,
                                             n_state_per_feature))
    relu_func = np.vectorize(ReLU)

    for episode in range(20000):
        state = env.reset()
        irl_score = 0.
        done = False

        if (episode != 0 and episode == 3000) or \
           (episode > 3000 and episode % 1500 == 0):
            learner = learner_feature_expectations / episode
            maxent_irl(expert, learner, theta, lr, relu_func)

        while not done:
            action = np.argmax(q_table[state[0]][state[1]])
            next_state, done = env.step(action)

            irl_reward = get_reward(theta, state)
            update_q_table(q_table, gamma, alpha, state, action, irl_reward,
                           next_state)
            learner_feature_expectations[state[0]][state[1]] += 1
            irl_score += irl_reward
            state = next_state

        print(f"Episode: {episode + 1}, Score: {irl_score}")

    np.save("q_table", q_table)
    print("Q Table saved")
    np.save("reward_parameter", theta)
    print("Reward Parameter saved")
