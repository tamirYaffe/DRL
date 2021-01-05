import time
import matplotlib.pyplot as plt
import os
import gym
import numpy as np
import sklearn.preprocessing
import tensorflow as tf
import collections

path_sep = os.path.sep
saved_models_dir = 'saved_models'
tf.compat.v1.disable_eager_execution()

# env = gym.make('CartPole-v1')
# env_state_size = 4
# env_action_size = 2

env = gym.make('MountainCarContinuous-v0')
# Continuous action space: (-1.000 to 1.000)
# Reward range: (-inf, inf)
# Observation range, dimension 0: (-1.200 to 0.600)
# Observation range, dimension 1: (-0.070 to 0.070)
env_state_size = 2
env_action_size = 1
env_action_space_low = env.action_space.low[0]
env_action_space_high = env.action_space.high[0]

np.random.seed(1)

# sample from state space for state normalization

state_space_samples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)
mean = np.mean(state_space_samples)
std = np.std(state_space_samples)


# function to normalize states
def normalize_state(state):  # requires input shape=(2,)
    scaled = scaler.transform([state])
    return scaled.reshape(-1)

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

            self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, 12],
                                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                            mode="fan_avg",
                                                                                                            distribution="uniform",
                                                                                                            seed=0))
            self.b1 = tf.compat.v1.get_variable("b1", [12], initializer=tf.compat.v1.zeros_initializer())
            self.W2 = tf.compat.v1.get_variable("W2", [12, 1],
                                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                            mode="fan_avg",
                                                                                                            distribution="uniform",
                                                                                                            seed=0))
            self.b2 = tf.compat.v1.get_variable("b2", [1], initializer=tf.compat.v1.zeros_initializer())

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
        self.init_xavier = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg",
                                                                           distribution="uniform")
        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.compat.v1.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.compat.v1.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.compat.v1.get_variable("W1", [self.state_size, 12],
                                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                            mode="fan_avg",
                                                                                                            distribution="uniform",
                                                                                                            seed=0))
            self.b1 = tf.compat.v1.get_variable("b1", [12], initializer=tf.compat.v1.zeros_initializer())
            self.W2 = tf.compat.v1.get_variable("W2", [12, self.action_size],
                                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                            mode="fan_avg",
                                                                                                            distribution="uniform",
                                                                                                            seed=0))
            self.b2 = tf.compat.v1.get_variable("b2", [self.action_size], initializer=tf.compat.v1.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.mu = tf.compat.v1.layers.dense(self.output, 1,
                                                None, self.init_xavier)
            self.sigma = tf.compat.v1.layers.dense(self.output, 1,
                                                   None, self.init_xavier)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5

            self.norm_dist = tf.compat.v1.distributions.Normal(self.mu, self.sigma)
            self.action = self.norm_dist.sample()
            self.action = tf.clip_by_value(self.action, env_action_space_low, env_action_space_high)

            self.loss = -tf.compat.v1.log(
                self.norm_dist.prob(
                    self.action) + 1e-5) * self.R_t - 1e-5 * self.norm_dist.entropy()
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


# Define hyperparameters
global_state_size = 6
global_action_size = 3

max_episodes = 5000
max_steps = 999
discount_factor = 0.99
# learning_rate = 0.0001
lr_actor = 0.0002  # set learning rates
lr_critic = 0.001

n_step = 20
render = False

# Initialize the policy network
tf.compat.v1.reset_default_graph()
policy = PolicyNetwork(global_state_size, global_action_size, lr_actor)
state_value_network = StateValueNetwork(global_state_size, lr_critic)


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

    # saver = tf.compat.v1.train.Saver()

    # load model
    loader = tf.compat.v1.train.import_meta_graph(saved_models_dir + path_sep + "actor_critic_model.meta")
    loader.restore(sess, saved_models_dir + path_sep + 'actor_critic_model')

    # reset output layer
    policy.W2 = tf.compat.v1.get_variable("policy_networkW2", [12, 1],
                                          initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                      mode="fan_avg",
                                                                                                      distribution="uniform",
                                                                                                      seed=0))
    policy.b2 = tf.compat.v1.get_variable("policy_networkb2", [1], initializer=tf.compat.v1.zeros_initializer())

    state_value_network.W2 = tf.compat.v1.get_variable("state_value_networkW2", [12, 1],
                                        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0,
                                                                                                    mode="fan_avg",
                                                                                                    distribution="uniform",
                                                                                                    seed=0))
    state_value_network.b2 = tf.compat.v1.get_variable("state_value_networkb2", [1],
                                                       initializer=tf.compat.v1.zeros_initializer())

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
        state = normalize_state(state)
        state = pad_state(state)
        state = state.reshape([1, global_state_size])

        episode_transitions = []

        for step in range(max_steps):

            action = sess.run(policy.action, {policy.state: state})

            next_state, reward, done, _ = env.step(action)

            # pad next state to size of global state size.
            next_state = normalize_state(next_state.reshape(-1))
            next_state = pad_state(next_state)
            next_state = next_state.reshape([1, global_state_size])

            # calculate approx_value = b(St)
            state_value_approx = sess.run(state_value_network.state_value,
                                          {state_value_network.state: state})

            # calculate approx_value = b(S't)
            if done:
                next_state_value_approx = 0
            else:
                next_state_value_approx = sess.run(state_value_network.state_value,
                                                   {state_value_network.state: next_state})

            if render:
                env.render()

            episode_rewards[episode] += reward

            # backpropagation
            target = reward + discount_factor * next_state_value_approx
            advantage = target - state_value_approx

            actor_feed_dict = {policy.state: state, policy.R_t: advantage,
                               policy.action: action}
            _, actor_loss = sess.run([policy.optimizer, policy.loss], actor_feed_dict)

            critic_feed_dict = {state_value_network.state: state,
                                state_value_network.R_t: target}
            _, critic_loss = sess.run([state_value_network.optimizer, state_value_network.loss],
                                      critic_feed_dict)

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                else:
                    average_rewards = np.mean(episode_rewards[:episode + 1])

                history.append(average_rewards)
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                if average_rewards > 89 and episode > 98:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            # save models
            # saver.save(sess, saved_models_dir + path_sep + "actor_critic_model")
            break


end_time = time.time()
print("total time to converge: {}".format(end_time - start_time))
plot_history(history)
