
from Arena.constants import *
from RafaelPlayer.DQN_constants import *
import os
import time
import random
from collections import deque
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from keras.models import Model
from keras.optimizers import (Adam, RMSprop)
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
        Permute, merge, multiply, Lambda, Reshape, TimeDistributed, LSTM, RepeatVector, Permute)
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

import argparse

from DQN.deeprl_prj.policy import *
from DQN.deeprl_prj.objectives import *
from DQN.deeprl_prj.preprocessors import *
from DQN.deeprl_prj.utils import *
from DQN.deeprl_prj.core import  *

REPLAY_MEMORY_SIZE = 50000 # how many last samples to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000 # minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64 # how many samples to use for training
UPDATE_TARGET_EVERY = 15 # number of terminal states
OBSERVATION_SPACE_VALUES = (SIZE_X, SIZE_Y, 3)
IS_TRAINING = True
MODEL_NAME = 'red_blue_16X32X512X9_2'


class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overrided. saves logs with our step number. otherwise every .fit() will start writing from 0th step
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided. we train for one batch only, no need to save anythings at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so wont close writer
    def on_train_end(self, _):
        pass

    # custom method for saving metrics
    # creats writer, write custom metrics and close writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        # with self.writer.as_default():
        #     for name, value in logs.items():
        #         with self.writer.as_default():
        #             tf.summary.scalar(name, value, step=index)
        #         self.step += 1
        #         self.writer.flush()

        # for name, value in logs.items():
        #     with self.writer:
        #         tf.summary.scalar(name, value)
        #         self.step += 1
        #         self.writer.flush()

        pass

class decision_maker_DQN_keras:
    def __init__(self, path_model_to_load=None):
        self._previous_stats = {}
        self._action = {}
        self._epsilon = epsilon
        self.model = None
        self.target_model = None

        self.is_training = IS_TRAINING
        self.numberOfSteps = 0
        self.burn_in = True

        self.episode_number = 0
        self.episode_loss = 0
        self.episode_target_value = 0

        self._Initialize_networks(path_model_to_load)


    def get_args(self):
        parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
        parser.add_argument('--env', default='shoot me if you can', help='small world')
        parser.add_argument('-o', '--output', default='./log/', help='Directory to save data to')
        parser.add_argument('--seed', default=0, type=int, help='Random seed')
        parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
        parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
        parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
        parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial exploration probability in epsilon-greedy')
        parser.add_argument('--final_epsilon', default=0.05, type=float, help='Final exploration probability in epsilon-greedy')
        parser.add_argument('--exploration_steps', default=1000000, type=int, help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')
        parser.add_argument('--num_samples', default=100000000, type=int, help='Number of training samples from the environment in training')
        parser.add_argument('--num_frames', default=4, type=int, help='Number of frames to feed to Q-Network')
        parser.add_argument('--frame_width', default=15, type=int, help='Resized frame width')
        parser.add_argument('--frame_height', default=15, type=int, help='Resized frame height')
        parser.add_argument('--replay_memory_size', default=1000000, type=int, help='Number of replay memory the agent uses for training')
        parser.add_argument('--target_update_freq', default=10000, type=int, help='The frequency with which the target network is updated')
        parser.add_argument('--train_freq', default=4, type=int, help='The frequency of actions wrt Q-network update')
        parser.add_argument('--save_freq', default=50000, type=int, help='The frequency with which the network is saved')
        parser.add_argument('--eval_freq', default=50000, type=int, help='The frequency with which the policy is evlauted')
        # parser.add_argument('--num_burn_in', default=1500, type=int, help='Number of steps to populate the replay memory before training starts')
        parser.add_argument('--num_burn_in', default=50000, type=int,
                            help='Number of steps to populate the replay memory before training starts')
        parser.add_argument('--load_network', default=False, action='store_true', help='Load trained mode')
        parser.add_argument('--load_network_path', default='', help='the path to the trained mode file')
        parser.add_argument('--net_mode', default='dqn', help='choose the mode of net, can be linear, dqn, duel')
        parser.add_argument('--max_episode_length', default = 10000, type=int, help = 'max length of each episode')
        parser.add_argument('--num_episodes_at_test', default = 20, type=int, help='Number of episodes the agent plays at test')
        parser.add_argument('--ddqn', default=False, dest='ddqn', action='store_true', help='enable ddqn')
        parser.add_argument('--train', default=True, dest='train', action='store_true', help='Train mode')
        parser.add_argument('--test', dest='train', action='store_false', help='Test mode')
        parser.add_argument('--no_experience', default=False, action='store_true', help='do not use experience replay')
        parser.add_argument('--no_target', default=False, action='store_true', help='do not use target fixing')
        parser.add_argument('--no_monitor', default=False, action='store_true', help='do not record video')
        parser.add_argument('--task_name', default='', help='task name')
        parser.add_argument('--recurrent', default=False, dest='recurrent', action='store_true', help='enable recurrent DQN')
        parser.add_argument('--a_t', default=False, dest='a_t', action='store_true', help='enable temporal/spatial attention')
        parser.add_argument('--global_a_t', default=False, dest='global_a_t', action='store_true', help='enable global temporal attention')
        parser.add_argument('--selector', default=False, dest='selector', action='store_true', help='enable selector for spatial attention')
        parser.add_argument('--bidir', default=False, dest='bidir', action='store_true', help='enable two layer bidirectional lstm')

        args = parser.parse_args()

        return args



    def _set_previous_state(self, state):
        self._previous_stats = state

    def _set_epsilon(self, input_epsilon):
        self._epsilon = input_epsilon

    def reset_networks(self):
        self._Initialize_networks()

    def _Initialize_networks(self, path_model_to_load = None):
        # load model
        if path_model_to_load !=None:
            self.model = load_model(path_model_to_load)
            self.target_model = load_model(path_model_to_load)
            self.target_model.set_weights(self.model.get_weights())

        else: #create new model
            args = self.get_args()

            self.num_actions = NUMBER_OF_ACTIONS
            input_shape = (args.frame_height, args.frame_width, args.num_frames)
            self.history_processor = HistoryPreprocessor(args.num_frames - 1)
            self.atari_processor = AtariPreprocessor()
            self.memory = ReplayMemory(args)
            self.policy = LinearDecayGreedyEpsilonPolicy(args.initial_epsilon, args.final_epsilon,
                                                         args.exploration_steps)
            self.gamma = args.gamma
            self.target_update_freq = args.target_update_freq
            self.num_burn_in = args.num_burn_in
            self.train_freq = args.train_freq
            self.batch_size = args.batch_size
            self.learning_rate = args.learning_rate
            self.frame_width = args.frame_width
            self.frame_height = args.frame_height
            self.num_frames = args.num_frames
            self.output_path = args.output
            self.output_path_videos = args.output + '/videos/'
            self.save_freq = args.save_freq
            self.load_network = args.load_network
            self.load_network_path = args.load_network_path
            self.enable_ddqn = args.ddqn
            self.net_mode = args.net_mode
            self.q_network = self.create_model(input_shape, self.num_actions, self.net_mode, args, "QNet")
            self.target_network = self.create_model(input_shape, self.num_actions, self.net_mode, args, "TargetNet")
            print(">>>> Net mode: %s, Using double dqn: %s" % (self.net_mode, self.enable_ddqn))
            self.eval_freq = args.eval_freq
            self.no_experience = args.no_experience
            self.no_target = args.no_target
            print(">>>> Target fixing: %s, Experience replay: %s" % (not self.no_target, not self.no_experience))

            # initialize target network
            self.target_network.set_weights(self.q_network.get_weights())
            self.final_model = None
            self.compile()

        # custom tesnsorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))


    def loadModel(self, model, target_model):
        # load existing models
        self.model = model
        self.target_model = target_model

    def create_model(self, input_shape, num_actions, mode, args, model_name='q_network'):
        """Create the Q-network model.

        Use Keras to construct a keras.models.Model instance.

        Parameters
        ----------
        window: int
          Each input to the network is a sequence of frames. This value
          defines how many frames are in the sequence.
        input_shape: tuple(int, int, int), rows, cols, channels
          The expected input image size.
        num_actions: int
          Number of possible actions. Defined by the gym environment.
        model_name: str
          Useful when debugging. Makes the model show up nicer in tensorboard.

        Returns
        -------
        keras.models.Model
          The Q-model.
        """
        assert (mode in ("linear", "duel", "dqn"))
        with tf.variable_scope(model_name):
            input_data = Input(shape=input_shape, name="input")
            if mode == "linear":
                flatten_hidden = Flatten(name="flatten")(input_data)
                output = Dense(num_actions, name="output")(flatten_hidden)
            else:
                if not (args.recurrent):
                    h1 = Convolution2D(32, (3, 3), strides=4, activation="relu", name="conv1")(input_data)
                    h2 = Convolution2D(64, (3, 3), strides=2, activation="relu", name="conv2")(h1)
                    # h3 = Convolution2D(64, (3, 3), strides = 1, activation = "relu", name = "conv3")(h2)
                    context = Flatten(name="flatten")(h2)
                else:
                    print('>>>> Defining Recurrent Modules...')
                    input_data_expanded = Reshape((input_shape[0], input_shape[1], input_shape[2], 1),
                                                  input_shape=input_shape)(input_data)
                    input_data_TimeDistributed = Permute((3, 1, 2, 4), input_shape=input_shape)(input_data_expanded)
                    h1 = TimeDistributed(Convolution2D(32, (8, 8), strides=4, activation="relu", name="conv1"), \
                                         input_shape=(args.num_frames, input_shape[0], input_shape[1], 1))(
                        input_data_TimeDistributed)
                    h2 = TimeDistributed(Convolution2D(64, (4, 4), strides=2, activation="relu", name="conv2"))(h1)
                    h3 = TimeDistributed(Convolution2D(64, (3, 3), strides=1, activation="relu", name="conv3"))(h2)
                    flatten_hidden = TimeDistributed(Flatten())(h3)
                    hidden_input = TimeDistributed(Dense(512, activation='relu', name='flat_to_512'))(flatten_hidden)
                    if not (args.a_t):
                        context = LSTM(512, return_sequences=False, stateful=False, input_shape=(args.num_frames, 512))(
                            hidden_input)
                    else:
                        if args.bidir:
                            hidden_input = Bidirectional(
                                LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)),
                                merge_mode='sum')(hidden_input)
                            all_outs = Bidirectional(
                                LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)),
                                merge_mode='sum')(hidden_input)
                        else:
                            all_outs = LSTM(512, return_sequences=True, stateful=False,
                                            input_shape=(args.num_frames, 512))(hidden_input)
                        # attention
                        attention = TimeDistributed(Dense(1, activation='tanh'))(all_outs)
                        # print(attention.shape)
                        attention = Flatten()(attention)
                        attention = Activation('softmax')(attention)
                        attention = RepeatVector(512)(attention)
                        attention = Permute([2, 1])(attention)
                        sent_representation = merge([all_outs, attention], mode='mul')
                        context = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(512,))(sent_representation)
                        # print(context.shape)

                if mode == "dqn":
                    h4 = Dense(512, activation='relu', name="fc")(context)
                    output = Dense(num_actions, name="output")(h4)
                elif mode == "duel":
                    value_hidden = Dense(512, activation='relu', name='value_fc')(context)
                    value = Dense(1, name="value")(value_hidden)
                    action_hidden = Dense(512, activation='relu', name='action_fc')(context)
                    action = Dense(num_actions, name="action")(action_hidden)
                    action_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keep_dims=True), name='action_mean')(
                        action)
                    output = Lambda(lambda x: x[0] + x[1] - x[2], name='output')([action, value, action_mean])
        model = Model(inputs=input_data, outputs=output)
        print(model.summary())
        return model


    def compile(self, optimizer = None, loss_func = None):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is the place to create the target network, setup
        loss function and any placeholders.
        """
        if loss_func is None:
            loss_func = mean_huber_loss
            # loss_func = 'mse'
        if optimizer is None:
            optimizer = Adam(lr = self.learning_rate)
            # optimizer = RMSprop(lr=0.00025)
        with tf.variable_scope("Loss"):
            state = Input(shape = (self.frame_height, self.frame_width, self.num_frames) , name = "states")
            action_mask = Input(shape = (self.num_actions,), name = "actions")
            qa_value = self.q_network(state)
            qa_value = merge([qa_value, action_mask], mode = 'mul', name = "multiply")
            qa_value = Lambda(lambda x: tf.reduce_sum(x, axis=1, keep_dims = True), name = "sum")(qa_value)

        self.final_model = Model(inputs = [state, action_mask], outputs = qa_value)
        self.final_model.compile(loss=loss_func, optimizer=optimizer)

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        state = state[None, :, :, :]
        return self.q_network.predict_on_batch(state)

    def train(self, new_state, reward, is_terminal):

        self.numberOfSteps += 1

        if is_terminal:
            # adding last frame only to save last state
            last_frame = self.atari_processor.process_state_for_memory(new_state)
            self.memory.append(last_frame, self._action, reward, is_terminal) #TODO in original code it was (last_frame, action, 0, is_terminal)- why 0?
            self.atari_processor.reset()
            self.history_processor.reset()
            if not self.burn_in:
                self.episode_reward = .0
                self.episode_raw_reward = .0
                self.episode_loss = .0
                self.episode_target_value = .0



        if not self.burn_in: # not enough samples in replay buffer
            if self.numberOfSteps % self.train_freq == 0:
                action_state = self.history_processor.process_state_for_network(
                    self.atari_processor.process_state_for_network(new_state))
                processed_reward = self.atari_processor.process_reward(reward)
                processed_next_state = self.atari_processor.process_state_for_network(new_state)
                action_next_state = np.dstack((action_state, processed_next_state))
                action_next_state = action_next_state[:, :, 1:]
                current_sample = Sample(action_state, self._action, processed_reward, action_next_state, is_terminal)
                loss, target_value = self.update_policy(current_sample)
                self.episode_loss += loss
                self.episode_target_value += target_value

            # update freq is based on train_freq
            if self.numberOfSteps % (self.train_freq * self.target_update_freq) == 0:
                # target updates can have the option to be hard or soft
                # related functions are defined in deeprl_prj.utils
                # here we use hard target update as default
                self.target_network.set_weights(self.q_network.get_weights())


            # if self.numberOfSteps % (self.eval_freq * self.train_freq) == 0:
            #     episode_reward_mean, episode_reward_std, eval_count = self.evaluate(env, 20, eval_count,
            #                                                                         max_episode_length, True)


        # max_future_q = np.max(self._Q_matrix[new_state]) # max Q value for this new observation
        # current_q = self._Q_matrix[self._previous_stats][self._action] # current Q for our chosen action
        #
        # if is_terminal:
        #     new_q = reward
        # else:
        #     new_q = (1- LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        #
        # self._Q_matrix[self._previous_stats][self._action] = new_q
        #
        # self._previous_stats = new_state
        self._previous_stats = new_state
        self.burn_in = (self.numberOfSteps < self.num_burn_in)

        if is_terminal:
            self.episode_number += 1

    def update_policy(self, current_sample):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        # current_sample = current_sample.img
        batch_size = self.batch_size

        if self.no_experience:
            states = np.stack([current_sample.state])
            next_states = np.stack([current_sample.next_state])
            rewards = np.asarray([current_sample.reward])
            mask = np.asarray([1 - int(current_sample.is_terminal)])

            action_mask = np.zeros((1, self.num_actions))
            action_mask[0, current_sample.action] = 1.0
        else:
            samples = self.memory.sample(batch_size)
            samples = self.atari_processor.process_batch(samples)

            states = np.stack([x.state for x in samples])
            actions = np.asarray([x.action for x in samples])
            action_mask = np.zeros((batch_size, self.num_actions))
            action_mask[range(batch_size), actions] = 1.0

            next_states = np.stack([x.next_state for x in samples])
            mask = np.asarray([1 - int(x.is_terminal) for x in samples])
            rewards = np.asarray([x.reward for x in samples])

        if self.no_target:
            next_qa_value = self.q_network.predict_on_batch(next_states)
        else:
            next_qa_value = self.target_network.predict_on_batch(next_states)

        if self.enable_ddqn:
            qa_value = self.q_network.predict_on_batch(next_states)
            max_actions = np.argmax(qa_value, axis = 1)
            next_qa_value = next_qa_value[range(batch_size), max_actions]
        else:
            next_qa_value = np.max(next_qa_value, axis = 1)
        target = rewards + self.gamma * mask * next_qa_value

        return self.final_model.train_on_batch([states, action_mask], target), np.mean(target)

    # adds step's data to memory replay array
    # (state, action, reward, new_state, is_terminal)
    def update_replay_memory(self, transition):
        self.memory.append(transition[0], transition[1], transition[2], transition[4])


    def _get_action(self, current_state, is_training = True, **kwargs):
        dqn_state = current_state.img
        """Select the action based on the current state.

        Returns
        --------
        selected action
        """
        self.numberOfSteps += 1
        policy_type = "UniformRandomPolicy" if self.burn_in else "LinearDecayGreedyEpsilonPolicy"
        state_for_network = self.atari_processor.process_state_for_network(dqn_state)
        action_state = self.history_processor.process_state_for_network(state_for_network)

        action = None
        q_values = self.calc_q_values(action_state) #shold be action_state
        if self.is_training:
            if policy_type == 'UniformRandomPolicy':
                action= UniformRandomPolicy(self.num_actions).select_action()
            else:
                # linear decay greedy epsilon policy
                action = self.policy.select_action(q_values, is_training)
        else:
            # return GreedyEpsilonPolicy(0.05).select_action(q_values)
            action = GreedyPolicy().select_action(q_values)

        self._epsilon = max([self._epsilon * EPSILONE_DECAY, 0.05])  # change epsilon
        self._action = action
        return action


# Agent class
class DQNAgent_keras:
    def __init__(self, path_model_to_load=None):
        self._previous_state = None
        self._action = None
        self.episode_number = 0
        self._decision_maker = decision_maker_DQN_keras(path_model_to_load)
        self.min_reward = -np.Inf
        self._type = AgentType.DQN_keras
        self.path_model_to_load = path_model_to_load

    def type(self) -> AgentType:
        return self._type

    def set_initial_state(self, initial_state_blue, episode_number):
        self.episode_number = episode_number
        self._previous_state = initial_state_blue
        self._decision_maker.tensorboard.step = episode_number
        pass

    def get_action(self, current_state):
        action = self._decision_maker._get_action(current_state)
        self._action = AgentAction(action)
        return self._action

    def update_context(self, new_state, reward, is_terminal):
        previous_state_for_network = self._decision_maker.atari_processor.process_state_for_memory(self._previous_state)
        new_state_for_network = self._decision_maker.atari_processor.process_state_for_memory(new_state)
        transition = (previous_state_for_network, self._action, reward, new_state_for_network, is_terminal)
        self._decision_maker.update_replay_memory(transition)
        # self._decision_maker.memory(transition)
        self._decision_maker.train(new_state, reward, is_terminal)
        self._previous_state = new_state

    def save_model(self, ep_rewards, path_to_model, player_color):

        avg_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        min_reward = min(ep_rewards[-SHOW_EVERY:])
        max_reward = max(ep_rewards[-SHOW_EVERY:])
        #TODO: uncomment this! # self._decision_maker.tensorboard.update_state(reward_avg = avg_reward, reward_min = min_reward, reward_max = max_reward, epsilon = epsilon)

        episode = len(ep_rewards)
        # save model, but only when min reward is greater or equal a set value
        if max_reward >=self.min_reward or episode == NUM_OF_EPISODES-1:
            self.min_reward = min_reward
            if player_color == Color.Red:
                color_str = "red"
            elif player_color == Color.Blue:
                color_str = "blue"
            self._decision_maker.q_network.save(
                f'{path_to_model+os.sep+MODEL_NAME}_{color_str}_{NUM_OF_EPISODES}_{max_reward: >7.2f}max_{avg_reward: >7.2f}avg_{min_reward: >7.2f}min__{int(time.time())}.model')

        return self.min_reward
