'''
code learned from:  https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/

'''
import numpy as np
#import keras.backend.tensorflow_backend as backend
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Input
from keras.optimizers import adam_v2
from keras.callbacks import TensorBoard
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import csv

DATA_SET = "./dataset/new_total.csv"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'test'
MIN_REWARD = -10  # For model save
MEMORY_FRACTION = 0.20
SHOW_EP_PROGRESS = False

# Environment settings
EPISODES = 10_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class UAV_Env:
    BATCH_SIZE = 100
    OBSERVATION_SPACE_VALUES = 9
    ACTION_SPACE_SIZE = 10


    def __init__(self):
        self.current_frame = []
        self.done = 0
        self.action = 0
        self.steps = 0
        self.packet_index = 0
        self.cum_reward = 0
        self.BATCH_SIZE = 100
        with open(DATA_SET) as file:
            self.data = list(csv.reader(file))

    def reset(self): # Reset environment, done after each episode
        with open(DATA_SET) as file:
            self.data = list(csv.reader(file))

        # Generate index for data
        self.packet_index = 0
        start = random.randint(0,(len(self.data)-self.BATCH_SIZE))
        end = start + self.BATCH_SIZE
        self.current_frame = self.data[start:end] # Take a random 10,000 concurrent samples
        state = self.current_frame[self.packet_index][0:9]
        self.state = np.array([float(i) for i in state]) # everything but class as np array for NN
            
        self.action = 0
        self.steps = 0
        self.classifier = self.current_frame[self.packet_index][9] # Last index of frame is class
        self.packet_index += 1
        self.done = 0
        self.test = 0
        self.cum_reward = 0
        return self.state

    def step(self, action): # Take action, return [reward, is_done, next_obs]
        action_rewards = {0: "benign", # Match actions with rewards
                    1: "portscan",
                    2: "ddos",
                    3: "bot",
                    4: "bruteforce",
                    5: "xss",
                    6: "sql",
                    7: "ftp",
                    8: "ssh",
                    9: "dos"}

        if (action == 0) and (self.classifier == "benign"): # Allows benign packet through
            reward = 5
        elif (action == 0) and (self.classifier != "benign"): # Allows malicious packet through
            reward = -10 # Heavy penalty for false negative
        elif action_rewards[action] == self.classifier: # Properly identified attack
            reward = 10
        else:
            reward = -5 # Misidentified attack, but still didn't allow it through

        self.cum_reward += reward
        self.packet_index += 1

        # Check if done
        if (self.packet_index == len(self.current_frame)-1): # Made it to end of data
            self.done = 1 # Add in additional reward?
        #elif self.cum_reward <= -2000: #"loss" condition, don't score super low
        #   self.done = 1
            
        else:
            # Get next state if not done
            self.steps += 1
            state = self.current_frame[self.packet_index][0:9]
            self.state = np.array([float(i) for i in state]) # everything but class as np array for NN
            #print(state)
            self.classifier = self.current_frame[self.packet_index][9]
            #print(self.classifier)

            if self.test == 0:
                self.test = 1
        
        return self.state, reward, self.done

    def render(self): # Some output for showing progress? 
        pass

env = UAV_Env()

class neural_net:
    def __init__(self):
        self.model = self.create_model() # Train
        self.target_model = self.create_model() # Target
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

    def create_model(self):
        model = Sequential()

        model.add(Input(shape=[env.OBSERVATION_SPACE_VALUES])) # 1 x 9 array
        model.add(Dense(64))
        model.add(Activation('relu'))

        #model.add(Dense(128))
        #model.add(Activation('relu'))

        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dropout(0.2))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=adam_v2.Adam(lr=0.001), metrics=['accuracy'])

        return model

    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return # Build up replay memory before starting

        #print("Replay memory built")
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) # Sample 64 states from replay memory

        cur_states = np.array([index[0] for index in minibatch]) # isolate states from replay memory
        #print(cur_states)
        cur_q_vals = self.model.predict(cur_states) # predict q-values

        # Get future states and predict q-values
        fut_states = np.array([index[3] for index in minibatch])
        fut_q_vals = self.target_model.predict(fut_states)

        X = []
        Y = []
        time1 = time.time()
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(fut_q_vals[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            cur_q = cur_q_vals[index]
            cur_q[action] = new_q

            X.append(current_state)
            Y.append(cur_q)


        self.model.fit(np.array(X), np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)#, callbacks=[self.tensorboard] if terminal_state else None)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1,*state.shape))[0]

agent = neural_net()
ep_rewards = []

for episode in tqdm(range(1,EPISODES + 1), ascii=True, unit='episodes'):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1

    current_state = env.reset()
   #print(current_state)

    done = False
    cur_prog = 0
    prev_prog = 0
    while not done:
        if SHOW_EP_PROGRESS:
            cur_prog = round(step/1_000, 2)
            if (cur_prog != prev_prog):
                os.system('clear')
                print("Episode Progress: ", cur_prog)
            prev_prog = cur_prog

        if np.random.random() > epsilon: # greedy decision
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0,env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        episode_reward += reward

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done,step)

        current_state = new_state
        step += 1

    print("REWARD: ", episode_reward)    
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
            

agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')