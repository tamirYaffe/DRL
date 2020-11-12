import random
from collections import deque
import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def get_deep_model(observation_space, action_space, learning_rate):
    action_space = action_space

    model = Sequential()
    model.add(Dense(32, input_shape=(observation_space,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    return model


def take_action(state, model, action_space, exploration_rate):
    # todo: change to epsilon greedy like in QLearning.
    if np.random.rand() < exploration_rate:
        return random.randrange(0, action_space)
    q_values = model.predict(state)
    return np.argmax(q_values[0])


def experience_replay(target_model, QValue_model, memory, batch_size, GAMMA, update_steps, c_steps):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, state_next, done in minibatch:
        Y = reward
        if not done:
            Y = (reward + GAMMA * np.amax(target_model.predict(state_next)[0]))
        Q_values = QValue_model.predict(state)
        Q_values[0][action] = Y
        QValue_model.fit(state, Q_values, verbose=0)

        # Every C steps, update target model with QValue model weights
        if update_steps == c_steps:
            target_model.set_weights(QValue_model.get_weights())
            update_steps = 0
        else:
            update_steps += 1

    return update_steps


def deep_learning(env, num_episodes,
                  max_memory=1000000,
                  exploration_rate=1.0,
                  c_steps=10,
                  GAMMA=0.95,
                  learning_rate=1e-3,
                  batch_size=16,
                  exploration_decay=0.995,
                  min_exploration=0.01):

    # Initialization
    episode = 0
    update_steps = 0
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    memory = deque(maxlen=max_memory)

    # We separate the network into two different networks:

    # One for computing the targets (using an older set of parameters 𝜃-)
    target_model = get_deep_model(observation_space, action_space, learning_rate)

    # One for predicting the q-value which is being updated every minibatch optimization (using parameters 𝜃)
    QValue_model = get_deep_model(observation_space, action_space, learning_rate)

    while episode < num_episodes:

        episode += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0

        while True:  # for each step of the episode

            step += 1
            # env.render()
            action = take_action(state, QValue_model, action_space, exploration_rate)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, observation_space])
            if not done:
                reward = reward
            # todo: check why reward is (-reward) if done?
            else:
                reward = -reward

            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) >= batch_size:
                update_steps = experience_replay(target_model, QValue_model, memory, batch_size, GAMMA, update_steps,
                                                 c_steps)
                # As the learning goes on alpha and epsilon should decayed to stabilize and exploit the learned policy.
                # todo: change decay method.
                exploration_rate *= exploration_decay
                exploration_rate = max(min_exploration, exploration_rate)

            if done:
                print(
                    "Run: " + str(episode) + ", exploration: " + str(exploration_rate) + ", score: " + str(
                        step))
                break


def main():
    env = gym.make('CartPole-v1')
    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_episodes = 200
    deep_learning(env, num_episodes)


if __name__ == '__main__':
    main()
