from collections import defaultdict
import numpy as np

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


def epsilonGreedyAction(Q, epsilon, num_actions, state):
    """
    Creates an epsilon-greedy policy based
    on a given Q-function and epsilon.
    """

    action_probabilities = np.ones(num_actions,
                                   dtype=float) * epsilon / num_actions

    best_action = np.argmax(Q[state])
    action_probabilities[best_action] += (1.0 - epsilon)
    # choose action according to
    # the probability distribution
    action = np.random.choice(np.arange(
        len(action_probabilities)),
        p=action_probabilities)
    return action


def qLearning(env, num_episodes, discount_factor=0.95,
              alpha=0.6, epsilon=0.3, decay_rate=0.95):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy.

    :param env: our environment, one of openai gym environments.
    :param num_episodes: number of episodes we wish to learn from
    :param discount_factor: use in order to reduce the value of future rewards.(should be 0< <1)
    :param alpha: learning rate, Higher alpha means you are updating your Q values in big steps.(should decay over time)
    :param epsilon: selecting random actions with epsilon probability.(should decay over time)
    :param decay_rate: The decay rate for alpha and epsilon. (should be 0<<1)
    :return: The learned Q-table and statistics about the learning process.
    """

    # 1. Create a lookup table containing the approximation of the Q-value for each state-action pair.
    # 2. Initialize the table with zeros.
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

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
    while episode < 5000:

        # As the learning goes on alpha and epsilon should decayed to stabilize and exploit the learned policy.
        alpha *= decay_rate
        epsilon *= decay_rate

        # sample action a.(sample actions using decaying 𝜀 − 𝑔𝑟𝑒𝑒𝑑𝑦)
        action = epsilonGreedyAction(Q, epsilon, env.action_space.n, state)

        # get next state s'.
        # take action and get reward, transit to next state
        next_state, reward, terminal, _ = env.step(action)
        step_count += 1

        # Update statistics
        stats['episode_rewards'][episode] += reward
        stats['episode_lengths'][episode] = step_count

        # if s' is terminal: (max of 100 steps per episode)
        if terminal or step_count == 100:
            target = reward
            # sample new initial state s'
            state = env.reset()
            step_count = 0
            episode += 1

        else:
            best_next_action = np.argmax(Q[next_state])
            target = reward + discount_factor * Q[next_state][best_next_action]

        Q[state][action] = (1-alpha) * Q[state][action] + alpha * target
        state = next_state

    return Q, stats
