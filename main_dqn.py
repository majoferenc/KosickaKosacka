import random
import requests

import tensorflow as tf
import numpy as np
import os
import shutil

BASE_URL = "http://169.51.194.78:31798/"

# Controls if we want load existing NN and start prediction/inference with it
PREDICT_MODE = False

# maximum steps of episode/iteration
MAX_EP_STEPS = 600
# maximum number of episodes
MAX_EPISODES = 500
LR_A = 1e-4  # learning rate for Actor, or simply 0.0001
LR_C = 1e-4  # learning rate for Critic, or simply 0.0001
# value of reward
GAMMA = 0.9
# Actor iteration
REPLACE_ITER_A = 800
# Critic iteration
REPLACE_ITER_C = 700
# capacity of memory buffer
MEMORY_CAPACITY = 2000
BATCH_SIZE = 16
VAR_MIN = 0.1

# state dimension is equal to sensors number
STATE_DIM = 1
# in our case action dimension is 1
ACTION_DIM = 1
# defines all posibilities of action
ACTION_BOUND = [0, 1, 2, 3]

done = False
result = {"sensors": None}
move = None
VALID_MOVES = ['Forward', 'Backward',  'TurnLeft', 'TurnRight']


# Disable Tensorflow eager execution
tf.compat.v1.disable_eager_execution()

# Memory storing all action moves of Actors NN
class Memory(object):
    def __init__(self, memory_size, input_dims):
        # memory capacity
        self.memory_size = memory_size
        # initializing memory with zeros
        self.memory_state = np.zeros((memory_size, input_dims))
        self.memory_counter = 0

    # rewrite memory chunk with new data
    def store_transition(self, state, actor, reward, state_):
        transition = np.hstack((state, actor, [reward], state_))
        index = self.memory_counter % self.memory_size  # replace the old memory with new memory
        self.memory_state[index, :] = transition
        self.memory_counter += 1

    # getting sample data from memory
    def sample_buffer(self, batch_size):
        assert self.memory_counter >= self.memory_size, 'Memory has not been fulfilled'
        indices = np.random.choice(self.memory_size, size=batch_size)
        return self.memory_state[indices, :]


# Q Learning model based on maximizing reward gained, q refers to the function tha the algo computes
# Main NN, takes action
class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        # training tensorflow session
        self.sess = sess
        self.a_dim = action_dim
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.action_bound = action_bound
        # training learning rate parameter
        self.lr = learning_rate

        with tf.compat.v1.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(STATE, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(STATE_, scope='target_net', trainable=False)

        self.e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    # definition of Actor NN
    def _build_net(self, s, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            # creating initial weights
            init_w = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=None)
            # creating initial biases
            init_b = tf.constant_initializer(0.001)
            # first NN layer, regular densely-connected NN layer, 100 neurons, using RELU(Rectified Linear Unit)
            # activation function, which defines, when neuron will activate
            net = tf.keras.layers.Dense(100, activation=tf.nn.relu,
                                        kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                        trainable=trainable)(s)
            # second NN layer, regular densely-connected NN layer, 20 neurons, using RELU
            net = tf.keras.layers.Dense(20, activation=tf.nn.relu,
                                        kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                        trainable=trainable)(net)
            with tf.compat.v1.variable_scope('a'):
                # last NN layer, will return final move set of actions, which will Actor take
                actions = tf.keras.layers.Dense(self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                                name='a', trainable=trainable)(net)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    # learn fuction definition, starting tensorflow session of training
    def learn(self, s):  # batch update
        self.sess.run(self.train_op, feed_dict={STATE: s})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.compat.v1.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    # getting only first single action for move of Actor
    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={STATE: s})[0]  # single action

    # setting Critic NN as Actors gradient layer
    def add_grad_to_graph(self, a_grads):
        with tf.compat.v1.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.compat.v1.variable_scope('A_train'):
            opt = tf.keras.optimizers.RMSprop(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

# all tensorflow placeholder variables
with tf.compat.v1.name_scope('STATE'):
    STATE = tf.compat.v1.placeholder(tf.float32, shape=[None, STATE_DIM], name='state')
with tf.compat.v1.name_scope('REWARD'):
    REWARD = tf.compat.v1.placeholder(tf.float32, [None, 1], name='reward')
with tf.compat.v1.name_scope('STATE_'):
    STATE_ = tf.compat.v1.placeholder(tf.float32, shape=[None, STATE_DIM], name='state_')


# Create TensorFlow Session
sess = tf.compat.v1.Session()

# Create actor and critic
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)


# Create memory
memory_replay_buffer = Memory(MEMORY_CAPACITY, input_dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.compat.v1.train.Saver()
path = './model'

if PREDICT_MODE:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.compat.v1.global_variables_initializer())


def train():
    random_exploration = 2.  # control exploration
    for ep in range(MAX_EPISODES):
        state = env.reset()
        ep_step = 0

        for t in range(MAX_EP_STEPS):
            # Added exploration noise
            actor_state = actor.choose_action(state)
            actor_state = np.clip(np.random.normal(actor_state, random_exploration), *ACTION_BOUND)  # add randomness
            # to action selection for exploration

            # send action of actor to grass cutter env
            # get state or sensor info, reward value and done varibe
            state_, reward, done = env.move(actor_state)
            # add move to memory
            memory_replay_buffer.store_transition(state, actor_state, reward, state_)

            # start learning after memory is full
            if memory_replay_buffer.memory_counter > MEMORY_CAPACITY:
                random_exploration = max([random_exploration * .9995, VAR_MIN])  # decay the action randomness
                b_M = memory_replay_buffer.sample_buffer(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                # trigger Actor learn function
                actor.learn(b_s)

            state = state_
            # increment step by one
            ep_step += 1

            if done or t == MAX_EP_STEPS - 1:
                print('Iteration:', ep,
                      '| Steps taken: %i' % int(ep_step),
                      '| Random Exploration: %.2f' % random_exploration
                      )
                break

    # Save model for future prediction
    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join(path, 'model.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=True)
    print("\n====> Saving trained model into: %s\n" % save_path)

def step(sessionid, move):
    reponse = requests.get(BASE_URL+"step/", params={"id": sessionid, "move": move})
    return reponse.json()


def initsession():
    response = requests.get(BASE_URL+"init/")
    print(response.json())
    return response.json()

sessionid = initsession()['id']

input("Visualization: " +BASE_URL+"visualize/" +
      sessionid+"\nPress Enter to continue...")
while not done:
    # little logic to not cross border or bump to obstacle
    validmoves_local=VALID_MOVES
    if result["sensors"] in ["Obstacle", "Border"]:
        if move == "Forward":
            validmoves_local=["Backward"]
        elif move == "Backward":
            validmoves_local=["Forward"]

    move=random.choice(validmoves_local)
      # add move to memory
    result=step(sessionid, move)
    done=result["done"]
    print(move, result)
