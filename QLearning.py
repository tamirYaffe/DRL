from collections import defaultdict
import numpy as np
import itertools

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
#       Qk+1(s, a) = (1- alpha) * Qk(s, a) + alpha * (target)
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


def qLearning(env, num_episodes, discount_factor=1.0,
              alpha=0.6, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy"""

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    # todo: check if its the right dict to use
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # we need to keep track on.(Q-table,
    #       reward per episode, average number of steps to the goal over last 100 episodes)
    stats = {'episode_lengths': np.zeros(num_episodes),
             'episode_rewards': np.zeros(num_episodes)}

    # For every episode
    for episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in itertools.count():

            # choose action according to epsilon greedy
            action = epsilonGreedyAction(Q, epsilon, env.action_space.n, state)

            # take action and get reward, transit to next state
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats['episode_rewards'][episode] += reward
            stats['episode_lengths'][episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            # done is True if episode terminated
            if done:
                break

            state = next_state

    return Q, stats
