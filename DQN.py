import os
import pickle
import random
from collections import deque
import gym
import numpy as np
from keras import Input, Sequential, Model
from keras.layers import Dense, Lambda, Activation
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

# for colab
path_separator = os.path.sep
drive_path = '/content/drive/My Drive/colab_DRL/'
saved_models_path = drive_path + "models" + path_separator


def get_dueling_deep_model(observation_space, action_space, learning_rate):
    input_layer = Input(observation_space)

    features = Dense(64, activation='relu')(input_layer)
    features = Dense(32, activation='relu')(features)

    values = Dense(16, activation='relu')(features)
    values = Dense(1)(values)

    advantages = Dense(16, activation='relu')(features)
    advantages = Dense(action_space)(advantages)

    # Add a customized layer to compute the absolute difference between the encodings
    aggregation_layer = Lambda(lambda tensors: tensors[0] + tensors[1] - K.mean(tensors[1]))
    aggregation = aggregation_layer([values, advantages])
    aggregation = Activation('linear')(aggregation)

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(action_space, activation='linear')(aggregation)

    # Connect the inputs with the outputs
    model = Model(inputs=input_layer, outputs=prediction)
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    return model


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
                      observation_space, is_DDQN=False):
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

    if not is_DDQN:
        predictions = target_model.predict(next_states)
        max_predictions = np.amax(predictions, axis=1)
        max_predictions = np.reshape(max_predictions, [batch_size, 1])

    else:
        # DDQN improvement
        target_predictions = target_model.predict(next_states)
        Qvalue_predictions = QValue_model.predict(next_states)
        Qvalue_max_predictions_indecies = np.argmax(Qvalue_predictions, axis=1)
        target_max_value_predictions = target_predictions[np.arange(len(target_predictions)), Qvalue_max_predictions_indecies]
        max_predictions = np.reshape(target_max_value_predictions, [batch_size, 1])

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
    # Initialization
    episode = 0
    update_steps = 0
    score_sum = 0
    stats = {'episode_loss': [],
             'episode_rewards': []}
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    memory = deque(maxlen=max_memory)

    # We separate the network into two different networks:

    # One for computing the targets (using an older set of parameters 𝜃-)
    # target_model = get_deep_model(observation_space, action_space, learning_rate)
    target_model = get_dueling_deep_model(observation_space, action_space, learning_rate)

    # One for predicting the q-value which is being updated every minibatch optimization (using parameters 𝜃)
    # QValue_model = get_deep_model(observation_space, action_space, learning_rate)
    QValue_model = get_dueling_deep_model(observation_space, action_space, learning_rate)

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
                # target_model.set_weights(QValue_model.get_weights())
                if episode > 100:
                    score_sum -= stats['episode_rewards'][episode - 100]
                score_sum += step
                print(
                    "episode: " + str(episode) + ", exploration: " + "{:.2f}".format(exploration_rate) + ", score: " + str(
                        step) + ", avg_score: " + "{:.2f}".format(score_sum / min(100, episode)))
                stats['episode_rewards'].append(step)

                # test model for early stopping
                if step >= 400:
                    model_score = test_model(env, 100, QValue_model)
                    if model_score >= 475:
                        return QValue_model, target_model, stats
                break

    return QValue_model, target_model, stats


def test_model(env, num_episodes, model):
    # Initialization
    episode = 0
    score_sum = 0
    observation_space = env.observation_space.shape[0]

    while episode < num_episodes:

        episode += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0

        while True:  # for each step of the episode

            step += 1
            # env.render()
            actions = model.predict(state)
            action = np.argmax(actions)

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, observation_space])
            state = next_state

            if done:
                score_sum += step
                print(
                    "episode: " + str(episode) + ", score: " + str(
                        step) + ", avg_score: " + "{:.2f}".format(score_sum / min(100, episode)))
                break
    return score_sum / min(100, episode)


def main():
    env = gym.make('CartPole-v1')
    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    num_episodes = 200
    QValue_model, target_model, stats = deep_learning(env, num_episodes, batch_size=16, c_steps=4, exploration_decay=0.05)

    # colab save model
    # target_model.save(saved_models_path + 'target_model.h5')
    # QValue_model.save(saved_models_path + 'QValue_model.h5')
    # with open(saved_models_path + 'stats.pickle', 'wb') as f:
    #     pickle.dump(stats, f)

    # local save model
    target_model.save('target_model.h5')
    QValue_model.save('QValue_model.h5')
    with open('stats.pickle', 'wb') as f:
        pickle.dump(stats, f)


def plot_stats():
    # read pickle file
    with open('stats.pickle', 'rb') as f:
        stats = pickle.load(f)

    plt.plot(stats['episode_loss'])
    plt.title('training steps loss')
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.show()

    stats['episode_rewards'] = stats['episode_rewards'][:61]
    plt.plot(stats['episode_rewards'])
    plt.title('episode rewards')
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()


if __name__ == '__main__':
    # main()
    plot_stats()
