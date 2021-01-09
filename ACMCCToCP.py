import time
import matplotlib.pyplot as plt
import os
import gym
import numpy as np
import tensorflow as tf
import collections
path_sep = os.path.sep
saved_models_dir = 'saved_models'
tf.compat.v1.disable_eager_execution()

env = gym.make('CartPole-v1')
env_state_size = 4
env_action_size = 2

np.random.seed(1)


# Rt is calculated as rewards and discount factor


def plot_history(history):
    plt.plot(history)
    plt.title('episode rewards')
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()


class StateValueNetwork:
    def __init__(self, state_size, learning_rate, name='state_value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.value = tf.compat.v1.placeholder(tf.int32, 1, name="value")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.compat.v1.get_variable("CP_W1", [self.state_size, 12],
                                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                            mode="fan_avg",
                                                                                                            distribution="uniform",
                                                                                                            seed=0))
            self.b1 = tf.compat.v1.get_variable("CP_b1", [12], initializer=tf.compat.v1.zeros_initializer())
            self.W2 = tf.compat.v1.get_variable("CP_W2", [12, 1],
                                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                            mode="fan_avg",
                                                                                                            distribution="uniform",
                                                                                                            seed=0))
            self.b2 = tf.compat.v1.get_variable("CP_b2", [1], initializer=tf.compat.v1.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # state value estimation
            self.state_value = tf.squeeze(self.output)
            # Loss
            self.loss = tf.reduce_mean(tf.math.squared_difference(self.state_value, self.R_t))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.compat.v1.get_variable("CP_W1", [self.state_size, 12],
                                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                            mode="fan_avg",
                                                                                                            distribution="uniform",
                                                                                                            seed=0))
            self.b1 = tf.compat.v1.get_variable("CP_b1", [12], initializer=tf.compat.v1.zeros_initializer())
            self.W2 = tf.compat.v1.get_variable("CP_W2", [12, self.action_size],
                                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                            mode="fan_avg",
                                                                                                            distribution="uniform",
                                                                                                            seed=0))
            self.b2 = tf.compat.v1.get_variable("CP_b2", [self.action_size], initializer=tf.compat.v1.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(input_tensor=self.neg_log_prob * self.R_t)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyperparameters
global_state_size = 6
global_action_size = 3

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.0004

n_step = 100
render = False

# Initialize the policy network
tf.compat.v1.reset_default_graph()
policy = PolicyNetwork(global_state_size, global_action_size, learning_rate)
state_value_network = StateValueNetwork(global_state_size, learning_rate)


def pad_state(state_to_pad):
    state_to_pad_size = len(state_to_pad)

    if state_to_pad_size == global_state_size:
        return state_to_pad

    padded_state = np.zeros(global_state_size)
    for i in range(state_to_pad_size):
        padded_state[i] = state_to_pad[i]

    return padded_state


def remove_actions_padding(actions):
    if len(actions) == env_action_size:
        return actions

    actions = actions[:env_action_size]
    # normalize remaining probabilities to 1.
    sum_of_probabilities = np.sum(actions)
    if sum_of_probabilities <= 0:
      actions = [0.5, 0.5]
    else:
      actions = actions / sum_of_probabilities

    return actions

# Start training the agent with REINFORCE algorithm
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver()

    # load model
    # loader = tf.compat.v1.train.import_meta_graph(saved_models_dir + path_sep + 'CP' + path_sep + "CP_model.meta")
    # loader.restore(sess, saved_models_dir + path_sep + 'CP' + path_sep + "CP_model")

    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state",
                                                       "state_value_approx",
                                                       "next_state_value_approx",
                                                       "done"])
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    start_time = time.time()
    history = []

    for episode in range(max_episodes):
        state = env.reset()

        # pad state to size of global state size.
        state = pad_state(state)

        state = state.reshape([1, global_state_size])
        episode_transitions = []

        for step in range(max_steps):

            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})

            # remove padding from actions
            # actions_distribution = remove_actions_padding(actions_distribution)

            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            if action >= env_action_size:
                next_state, reward, done = state, 0, True
            else:
                next_state, reward, done, _ = env.step(action)
                # pad next state to size of global state size.
                next_state = pad_state(next_state)
                next_state = next_state.reshape([1, global_state_size])

            # calculate approx_value = b(St)
            state_value_approx = sess.run(state_value_network.state_value, {state_value_network.state: state})

            # calculate approx_value = b(S't)
            if done:
                next_state_value_approx = 0
            else:
                next_state_value_approx = sess.run(state_value_network.state_value,
                                                   {state_value_network.state: next_state})

            if render:
                env.render()

            action_one_hot = np.zeros(global_action_size)
            action_one_hot[action] = 1
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward,
                                                  next_state=next_state,
                                                  state_value_approx=state_value_approx,
                                                  next_state_value_approx=next_state_value_approx, done=done))
            episode_rewards[episode] += reward

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                else:
                    average_rewards = np.mean(episode_rewards[:episode + 1])

                history.append(average_rewards)
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > 475 and episode > 98:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            # save models
            saver.save(sess, saved_models_dir + path_sep + 'CP' + path_sep + "CP_model")
            break

        # Compute Rt for each time-step t and update the network's weights
        for t, transition in enumerate(episode_transitions):
            # total_discounted_return = sum(discount_factor ** i * t.reward for i, t in
            #                               enumerate(episode_transitions[t:]))  # Rt

            # total_approx_discounted_return = transition.reward + sum(discount_factor ** (i+1)
            #                                                          * t.next_state_value_approx for i, t in
            #                                                          enumerate(episode_transitions[t:]))

            total_discounted_return = sum(discount_factor ** i * t.reward for i, t in
                                          enumerate(episode_transitions[t:t + n_step-1] if t + n_step-1 < len(episode_transitions)
                                                    else episode_transitions[t:]))  # Rt

            if t + n_step-1 < len(episode_transitions):
                total_approx_discounted_return = discount_factor ** n_step *\
                                                 episode_transitions[t + n_step-1].next_state_value_approx
                total_discounted_return += total_approx_discounted_return

            # total_approx_discounted_return = transition.reward + discount_factor * transition.next_state_value_approx

            state_value_approx = transition.state_value_approx  # b(st)

            advantage = total_discounted_return - state_value_approx

            actor_feed_dict = {policy.state: transition.state, policy.R_t: advantage,
                               policy.action: transition.action}
            _, actor_loss = sess.run([policy.optimizer, policy.loss], actor_feed_dict)

            critic_feed_dict = {state_value_network.state: transition.state,
                                state_value_network.R_t: total_discounted_return}
            _, critic_loss = sess.run([state_value_network.optimizer, state_value_network.loss],
                                      critic_feed_dict)
end_time = time.time()
print("total time to converge: {}".format(end_time - start_time))
plot_history(history)
