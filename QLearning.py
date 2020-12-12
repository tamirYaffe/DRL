import random
import gym
import numpy as np
import matplotlib.pyplot as plt


# Algorithm
# 1. Create a lookup table containing the approximation of the Q-value for each state-action pair.
# 2. Initialize the table with zeros.
# 3. Choose initial values for the hyper-parameters: learning rate 𝛼, discount
#    factor 𝛾, decay rate for decaying epsilon-greedy probability.
# 4. Get initial state s
# 5. For k = 1, 2, ... till convergence (5000 episodes)
#       sample action a, get next state s' (sample actions using decaying 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦)
#       if s' is terminal: (max of 100 steps per episode)
#           target = R(s, a, s')
#           sample new initial state s'
#       else:
#           target = R(s, a, s') + ymaxQk(s', a')
#       Qk+1(s, a) = (1- 𝛼) * Qk(s, a) + 𝛼 * (target)
#       s = s'
from matplotlib.colors import ListedColormap


def epsilonGreedyAction(q_table, epsilon, state):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.
    """
    actions = q_table[state]
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(actions)
    else:
        action = random.randrange(len(actions))
    return action


def qLearning(env, num_episodes, discount_factor=0.99,
              min_alpha=0.01, max_alpha=0.8, min_epsilon=0.01, max_epsilon=1, decay_rate=0.001):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy.

    :param env: our environment, one of openai gym environments.
    :param num_episodes: number of episodes we wish to learn from
    :param discount_factor: use in order to reduce the value of future rewards.(should be 0< <1)
    :param min_alpha: min value of learning rate, Higher alpha means you are updating your Q values in big steps.
    :param max_alpha: starting value of learning rate, Higher alpha means you are updating your Q values in big steps.
    :param min_epsilon: min value for selecting random actions with epsilon probability.
    :param max_epsilon: starting value for selecting random actions with epsilon probability.
    :param decay_rate: The decay rate for alpha and epsilon. (should be 0<<1)
    :return: The learned Q-table and statistics about the learning process.
    """
    alpha = max_alpha
    epsilon = max_epsilon

    # 1. Create a lookup table containing the approximation of the Q-value for each state-action pair.
    # 2. Initialize the table with zeros.
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n
    q_table = np.zeros((state_space_size, action_space_size))
    start_q_table = []
    middle_q_table = []

    # 3. Choose initial values for the hyper-parameters: learning rate 𝛼, discount
    #    factor 𝛾, decay rate for decaying epsilon-greedy probability.

    # we need to keep track on.(Q-table,
    #       reward per episode, average number of steps to the goal over last 100 episodes)
    stats = {'episode_lengths': np.zeros(num_episodes),
             'episode_rewards': np.zeros(num_episodes)}

    # 4. Get initial state s
    state = env.reset()
    step_count = 0
    episode = 1

    # 5. For k = 1, 2, ... till convergence (5000 episodes)
    while episode < num_episodes:

        # sample action a.(sample actions using decaying 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦)
        action = epsilonGreedyAction(q_table, epsilon, state)

        # get next state s'.
        # take action and get reward, transit to next state
        next_state, reward, done, _ = env.step(action)
        step_count += 1

        # Update statistics
        stats['episode_rewards'][episode] += reward
        stats['episode_lengths'][episode] = step_count

        # if s' is terminal: (max of 100 steps per episode)
        if done or step_count == 100:
            target = reward

        else:
            best_next_action = np.argmax(q_table[next_state])
            target = reward + discount_factor * q_table[next_state][best_next_action]
            # target = reward + discount_factor * np.max(q_table[next_state, :])

        q_table[state][action] = (1-alpha) * q_table[state][action] + alpha * target
        state = next_state

        if done or step_count == 100:

            if done and reward == 0:
                stats['episode_lengths'][episode] = 100

            # save q_tables
            if episode == 500:
                start_q_table = np.copy(q_table)
            if episode == 2000:
                middle_q_table = np.copy(q_table)
            # sample new initial state s'
            state = env.reset()
            step_count = 0
            episode += 1

            # As the learning goes on alpha and epsilon should decayed to stabilize and exploit the learned policy.
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
            alpha = min_alpha + (max_alpha - min_alpha) * np.exp(-decay_rate * episode)

    return start_q_table, middle_q_table, q_table, stats


def plot_colormap(colormaps, data):
    """
    Helper function to plot data with associated colormap.
    """
    np.random.seed(19680801)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=False, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=data.min(), vmax=data.max())
        fig.colorbar(psm, ax=ax)
    plt.show()


def plot_results(middle_q_table, num_episodes, q_table, start_q_table, stats):
    length_per_hundred_episodes = np.split(stats['episode_lengths'], num_episodes / 100)
    count = 100
    length_sum_per_hundred_episodes = []
    for length in length_per_hundred_episodes:
        length_sum_per_hundred_episodes.append(sum(length / 100))
        # print(count, ": ", str(sum(length / 100)))
        count += 100
    plt.plot(length_sum_per_hundred_episodes)
    plt.title('average number of steps to the goal over last 100')
    plt.ylabel('average length')
    plt.xlabel('hundredth episodes')
    plt.show()
    cmap = ListedColormap(["khaki", "gold", "orange", "darkorange", "orangered", "red", "firebrick"])
    plot_colormap([cmap], start_q_table)
    plot_colormap([cmap], middle_q_table)
    plot_colormap([cmap], q_table)


def main():
    env = gym.make('FrozenLake-v0')
    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_episodes = 5000
    start_q_table, middle_q_table, q_table, stats = qLearning(env, num_episodes)

    plot_results(middle_q_table, num_episodes, q_table, start_q_table, stats)


if __name__ == '__main__':
    main()
