import time
import random
import requests
import webbrowser, pyautogui
import argparse
import tensorflow as tf
import numpy as np
import os
import shutil
import api_util as api

cli = argparse.ArgumentParser()
cli.add_argument("--maps", nargs="+", default=["SimpleMap", "Greenland", "Localhost", "PuertoRico", "Oregon", "Safari"],)
cli.add_argument("--base_url", nargs="+", default=["http://localhost/"])
cli.add_argument("--render_mode", nargs="+", default=["False"])
cli.add_argument("--predict_mode", nargs="+", default=["False"])
args = cli.parse_args()

print("maps: %r" % args.maps)
print("base_url: %s" % str(args.base_url))
print("render_mode: %s" % str(args.render_mode))
print("predict_mode: %s" % str(args.predict_mode))

done = False
result = {"sensors": None}
move = None
VALID_MOVES = ['Forward', 'Backward',  'TurnLeft', 'TurnRight']

# Check for --version or -V
if args.render_mode[0] == "True":
    print("Render mode is enabled.")
    RENDER_MODE = True
else:
    RENDER_MODE = False

if args.predict_mode[0] == "True":
    print("Predict mode is enabled.")
    PREDICT_MODE = True
else:
    PREDICT_MODE = False

time.sleep(10)

# maximum number of episodes for one map
MAX_EPISODES = 2000
LR_A = 1e-4  # learning rate for Actor, or simply 0.0001
LR_C = 1e-4  # learning rate for Critic, or simply 0.0001
# Actor iteration
REPLACE_ITER_A = 800
# Critic iteration
REPLACE_ITER_C = 700
# capacity of memory buffer
MEMORY_CAPACITY = 20_000
BATCH_SIZE = 128
VAR_MIN = 0.1

# state dimension is equal to sensors number
STATE_DIM = 1
# in our case action dimension is 1
ACTION_DIM = 1
# defines all posibilities of action
ACTION_BOUND = [0,3]

# Disable Tensorflow eager execution
tf.compat.v1.disable_eager_execution()

def get_valid_move(dqn_answer):
    dqn_answer_rounded = int(dqn_answer)
    return {
        0: 'Forward',
        1: 'Backward',
        2: 'TurnLeft',
        3: 'TurnRight'
    }.get(dqn_answer_rounded, 'Forward')    # Forward is default if x is not found

def get_input_to_dqn_fron_sensors(sensor_value):
    return {
        "Obstacle": np.array([1.0]),
        "Border": np.array([2.0]),
        "Cut": np.array([3.0]),
        "OutOfBondaries": np.array([4.0]),
        "Stuck": np.array([5.0]),
        "Charged": np.array([6.0])
    }.get(sensor_value, np.array([7.0]))    # Forward is default if x is not found

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
                actions = tf.keras.layers.Dense(self.a_dim, activation=None, kernel_initializer=init_w,
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


# Second NN, inform Actor how good was the action taken and how it should adjust
class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, t_replace_iter, a, a_):
        # tensorflow session
        self.sess = sess
        # state dimension
        self.s_dim = state_dim
        # action dimension
        self.a_dim = action_dim
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.compat.v1.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self.build_dqn(STATE, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self.build_dqn(STATE_, a_, 'target_net',
                                     trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                        scope='Critic/eval_net')
            self.t_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                                        scope='Critic/target_net')

        with tf.compat.v1.variable_scope('target_q'):
            self.target_q = REWARD * self.q_

        with tf.compat.v1.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(input_tensor=tf.math.squared_difference(self.target_q, self.q))

        with tf.compat.v1.variable_scope('C_train'):
            self.train_op = tf.compat.v1.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        with tf.compat.v1.variable_scope('a_grad'):
            self.a_grads = tf.gradients(ys=self.q, xs=a)[0]  # tensor of gradients of each sample (None, a_dim)

    def build_dqn(self, s, a, scope, trainable):
        with tf.compat.v1.variable_scope(scope):
            # creating initial weights
            init_w = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", seed=None)
            # creating initial biases
            init_b = tf.constant_initializer(0.01)

            with tf.compat.v1.variable_scope('l1'):
                n_l1 = 100
                w1_s = tf.compat.v1.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.compat.v1.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.compat.v1.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            # first NN layer, regular densely-connected NN layer, 20 neurons, using RELU
            net = tf.keras.layers.Dense(20, activation=tf.nn.relu,
                                        kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                        trainable=trainable)(net)
            with tf.compat.v1.variable_scope('q'):
                q = tf.keras.layers.Dense(1, kernel_initializer=init_w, bias_initializer=init_b,
                                          trainable=trainable)(net)  # Q(s,a)
        return q

    def learn(self, state, action, reward, state_):
        self.sess.run(self.train_op, feed_dict={STATE: state, self.a: action, REWARD: reward, STATE_: state_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.compat.v1.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


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
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, REPLACE_ITER_C, actor.a, actor.a_)
# Setting Critic as Actors gradient layer
actor.add_grad_to_graph(critic.a_grads)

# Create memory
memory_replay_buffer = Memory(MEMORY_CAPACITY, input_dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.compat.v1.train.Saver()
path = './model'

if PREDICT_MODE:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.compat.v1.global_variables_initializer())

def train():
    for map in args.maps:
        print("map: %s" % map)
        # maximum steps of episode/iteration
        MAX_EP_STEPS = 500 # is overwriten by environment
        done = False
        status_code = 202
        result = {"sensors": None}
        backward_step_result =  {"sensors": None}
        real_step_result =  {"sensors": None}
        move = None
        VALID_MOVES = ['Forward', 'Backward',  'TurnLeft', 'TurnRight']
        random_exploration = 2.  # control exploration
        for ep in range(MAX_EPISODES):
            ep_step = 0
            done = False
            for t in range(MAX_EP_STEPS):
                session, status_code = api.init_session(map, args.base_url[0])
                if status_code > 202:
                        break
                MAX_EP_STEPS = session['stepsLimit']
                print(session["id"])
                if RENDER_MODE:
                    webbrowser.open(args.base_url[0]+"visualize/" + session["id"])
                while not done:
                    print("Episode: " + str(ep))
                    # little logic to not cross border or bump to obstacle
                    validmoves_local=VALID_MOVES
                    if result["sensors"] in ["Obstacle", "Border"]:
                        if move == "Forward":
                            validmoves_local=["Backward"]
                        elif move == "Backward":
                            validmoves_local=["Forward"]
                    if result["sensors"]:
                        print("Sensors:" + result["sensors"])
                        state = get_input_to_dqn_fron_sensors(result["sensors"])
                    else:
                        state = np.array([3.0])
                    # Added exploration noise
                    actor_state = actor.choose_action(state)
                    actor_state = np.clip(np.random.normal(actor_state, random_exploration), *ACTION_BOUND)  # add randomness
                    # send action of actor to grass cutter env
                    # get state or sensor info, reward value and done varibe
                    move = get_valid_move(actor_state)
                    # Create front sensor data:
                    result, status_code = api.step(session['id'], 'Forward', args.base_url[0])
                    
                    if status_code > 202:
                        ep_step += 1
                        break
                    
                    if result["sensors"] in ["OutOfBondaries", "Stuck"]:
                        print("====> Session ended: " + result["sensors"])
                        if RENDER_MODE:
                            ep_step += 1
                            pyautogui.hotkey('ctrl', 'w')
                        break

                    backward_step_result, status_code= api.step(session['id'], 'Backward', args.base_url[0])
                    
                    if status_code > 202:
                        ep_step += 1
                        break
                    
                    if backward_step_result["sensors"] in ["OutOfBondaries", "Stuck"]:
                        print("====> Session ended: " + backward_step_result["sensors"])
                        if RENDER_MODE:
                            ep_step += 1
                            pyautogui.hotkey('ctrl', 'w')
                        break

                    if 'done' in result:
                        done=result['done']
                    if 'done' in backward_step_result:
                        done=backward_step_result['done']

                    real_step_result, status_code = api.step(session['id'], move, args.base_url[0])

                    if status_code > 202:                
                        ep_step += 1
                        break

                    if 'done' in real_step_result:
                        done=real_step_result['done']
                    else:
                        ep_step += 1
                        break
                    reward = real_step_result["reward"]
                    # change reward mechanism
                    if real_step_result["reward"] == 0:
                        reward = -1
                    else:
                        reward = 0
                        
                    state_ = get_input_to_dqn_fron_sensors(result["sensors"])
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

                        # trigger Critic learn function
                        critic.learn(b_s, b_a, b_r, b_s_)
                        # trigger Actor learn function
                        actor.learn(b_s)

                    state = state_
                    # increment step by one
                    ep_step += 1
                    print(move, result)

                    if real_step_result["sensors"] in ["OutOfBondaries", "Stuck"]:
                        print("====> Session ended: " + real_step_result["sensors"])
                        ep_step += 1
                        if RENDER_MODE:
                            pyautogui.hotkey('ctrl', 'w')
                        break
                    if done is True:
                        print("====> Session ended: " + real_step_result["sensors"])
                        ep_step += 1
                        if RENDER_MODE:
                            pyautogui.hotkey('ctrl', 'w')
                        break

                if done or t == MAX_EP_STEPS - 1 or status_code > 202 or result["sensors"] in ["OutOfBondaries", "Stuck"] or real_step_result["sensors"] in ["OutOfBondaries", "Stuck"] or backward_step_result["sensors"] in ["OutOfBondaries", "Stuck"]:
                    ep_step += 1
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


def predict():
    for map in args.maps:
        print("map: %s" % map)
        while True:
            state = np.array([3.0])
            done = False
            result = {"sensors": None}
            move = None
            VALID_MOVES = ['Forward', 'Backward',  'TurnLeft', 'TurnRight']
            session, status_code =  api.init_session(map, args.base_url[0])
            if status_code > 202:
                break
            print(session["id"])
            if RENDER_MODE:
                webbrowser.open(args.base_url[0]+"visualize/" + session["id"])
            while True:
                if result["sensors"]:
                    print("Sensors:" + result["sensors"])
                    state = get_input_to_dqn_fron_sensors(result["sensors"])
                else:
                    state = np.array([3.0])
                actor_state = actor.choose_action(state)
                move = get_valid_move(actor_state)
                print("Move" + move)

                # Create front sensor data:
                result, status_code = api.step(session['id'], 'Forward', args.base_url[0])
                if status_code > 202:
                        break
                backward_step_result, status_code = api.step(session['id'], 'Backward', args.base_url[0])
                if status_code > 202:
                        break
                # Do real move
                real_result, status_code = api.step(session['id'], move, args.base_url[0])
                if status_code > 202:
                        break
                done=real_result["done"]
                # change reward mechanism
                if result["reward"] == 0:
                    reward = -1
                else:
                    reward = 0
                print(move, real_result)
                state_ = get_input_to_dqn_fron_sensors(result["sensors"])
                if real_result["sensors"] in ["OutOfBondaries", "Stuck"]:
                    print("Session ended: " + real_result["sensors"])
                    if RENDER_MODE:
                        pyautogui.hotkey('ctrl', 'w')
                    break
                state = state_
                if done:
                    break
            if done:
                break

if PREDICT_MODE:
    predict()
else:
    train()
    