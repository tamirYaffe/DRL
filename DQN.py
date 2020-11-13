import os
import pickle
import random
from collections import deque
import gym
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# for colab
path_separator = os.path.sep
drive_path = '/content/drive/My Drive/colab_DRL/'
saved_models_path = drive_path + "models" + path_separator


def get_deep_model(observation_space, action_space, learning_rate):
    model = Sequential()
    model.add(Dense(64, input_shape=(observation_space,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    return model


def epsilonGreedyAction(state, exploration_rate, model):
    actions = model.predict(state)
    if np.random.rand() > exploration_rate:
        action = np.argmax(actions)
    else:
        action = random.randrange(len(actions[0]))
    return action


def experience_replay(target_model, QValue_model, memory, batch_size, discount_factor, update_steps, c_steps,
                      observation_space):
    minibatch = random.sample(memory, batch_size)

    states = np.empty([batch_size, observation_space])
    actions = np.empty([batch_size, 1])
    rewards = np.empty([batch_size, 1])
    next_states = np.empty([batch_size, observation_space])
    dones = np.empty([batch_size, 1])
    for i in range(batch_size):
        states[i] = minibatch[i][0]
        actions[i] = minibatch[i][1]
        rewards[i] = minibatch[i][2]
        next_states[i] = minibatch[i][3]
        dones[i] = minibatch[i][4]

    predictions = target_model.predict(next_states)
    max_predictions = np.amax(predictions, axis=1)
    max_predictions = np.reshape(max_predictions, [batch_size, 1])
    Y = (rewards + discount_factor * max_predictions)
    Q_values = QValue_model.predict(states)

    for i in range(batch_size):
        if dones[i]:
            Q_values[i][int(actions[i])] = rewards[i]
        else:
            Q_values[i][int(actions[i])] = Y[i]

    history = QValue_model.fit(states, Q_values, verbose=0)

    # Every C steps, update target model with QValue model weights
    if update_steps == c_steps:
        target_model.set_weights(QValue_model.get_weights())
        update_steps = 0
    else:
        update_steps += 1

    return update_steps, history


def deep_learning(env, num_episodes,
                  batch_size=16,
                  c_steps=10,
                  exploration_decay=0.01,
                  max_memory=500000,
                  exploration_rate=1.0,
                  min_exploration=0.01,
                  max_exploration=1,
                  discount_factor=0.95,
                  learning_rate=1e-3):
    # todo:  Add stats records
    # Initialization
    episode = 0
    update_steps = 0
    score_sum = 0
    stats = {'episode_loss': [],
             'episode_rewards': np.zeros(num_episodes + 1)}
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
            action = epsilonGreedyAction(state, exploration_rate, QValue_model)
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
                update_steps, history = experience_replay(target_model, QValue_model, memory, batch_size,
                                                          discount_factor, update_steps,
                                                          c_steps, observation_space)
                stats['episode_loss'].append(history.history['loss'])
                # As the learning goes on alpha and epsilon should decayed to stabilize and exploit the learned policy.

                # decay method 1
                # exploration_decay = 0.995
                # exploration_rate *= exploration_decay
                # exploration_rate = max(min_exploration, exploration_rate)

                # decay method 2
                # exploration_decay = 0.001
                exploration_rate = min_exploration + (max_exploration - min_exploration) * np.exp(-exploration_decay *
                                                                                                  episode)

            if done:
                if episode > 100:
                    score_sum -= stats['episode_rewards'][episode - 100]
                score_sum += step
                print(
                    "episode: " + str(episode) + ", exploration: " + "{:.2f}".format(exploration_rate) + ", score: " + str(
                        step) + ", avg_score: " + "{:.2f}".format(score_sum / min(100, episode)))
                stats['episode_rewards'][episode] = step
                break
    return target_model, stats


def main():
    env = gym.make('CartPole-v1')
    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_episodes = 200
    target_model, stats = deep_learning(env, num_episodes, batch_size=16, c_steps=4, exploration_decay=0.05)

    # colab save model
    # target_model.save(saved_models_path + 'target_model.h5')
    # with open(saved_models_path + 'stats.pickle', 'wb') as f:
    #     pickle.dump(stats, f)

    # local save model
    target_model.save('target_model.h5')
    with open('stats.pickle', 'wb') as f:
        pickle.dump(stats, f)


if __name__ == '__main__':
    main()
