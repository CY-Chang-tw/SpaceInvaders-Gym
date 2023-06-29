import gym
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
import platform

gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

class DQN:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9
        self.epsilon = 0.01
        self.epsilon_min = 0.2
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.batch_size = 64

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=self.observation_space))
        model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        state, _ = state  # only keep the frame
        next_state, _ = next_state  # only keep the frame
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

EPISODES = 600

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
state_size = (210, 160, 3)
action_size = 6
agent = DQN(state_size, action_size)

if os.path.exists('dqn.h5'):
    agent.load('dqn.h5')

for e in range(EPISODES):
    state = env.reset()

    state = np.reshape(state[0], [1, state_size[0], state_size[1], state_size[2]])
    score = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
        agent.remember((state, info), action, reward, (next_state, info), done)
        score += reward
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e+1, EPISODES, score, agent.epsilon))
            break
    agent.replay()
    print("Saving model")
    agent.save('dqn.h5')


OBSERV_episo = 3
for e in range(OBSERV_episo):
    state = env.reset()
    state = np.reshape(state[0], [1, state_size[0], state_size[1], state_size[2]])
    score = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size[0], state_size[1], state_size[2]])
        score += reward
        state = next_state
        if done:
            print("Observation episode: {}/{}, score: {}".format(e + 1, OBSERV_episo, score))
            break

env.close()
