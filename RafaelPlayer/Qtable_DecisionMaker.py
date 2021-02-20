from Arena.CState import State
from Arena.AbsDecisionMaker import AbsDecisionMaker
from RafaelPlayer.QPlayer_constants import *
from Arena.constants import *
import numpy as np
import os
import pickle


class Qtable_DecisionMaker(AbsDecisionMaker):

    def __init__(self, UPDATE_CONTEXT=True , path_model_to_load=None):

        self._previous_state = {}
        self._action = -1
        self._epsilon = epsilon
        self._type = AgentType.Q_table
        self.episode_number = 0
        self.path_model_to_load = path_model_to_load
        self.UPDATE_CONTEXT = UPDATE_CONTEXT
        self.IS_TRAINING = IS_TRAINING

        if path_model_to_load is not None:
            p = path.join(RELATIVE_PATH_HUMAN_VS_MACHINE_DATA, path_model_to_load)
            self._Q_matrix = pickle.load(open(p, "rb"))
        else:
            self._Q_matrix = self.init_q_table()

    def set_initial_state(self, state: State, episode_number, input_epsilon=0.5):

        state_entry = (state.my_pos.get_tuple(), state.enemy_pos.get_tuple())
        self._previous_state = state_entry
        self._epsilon = input_epsilon
        self.episode_number = episode_number

    def init_q_table(self, start_q_table=None):
        if start_q_table is None:
            # Initialize the q-table
            # (x1, y1)- blue cor (if Entity is Blue, otherwise (x1, y1) is the red cor)
            # (x2, y2)- red cor (if Entity is Blue, otherwise (x1, y1) is the red cor)
            q_table = {}
            for x1 in range(0, SIZE_X):
                for y1 in range(0, SIZE_Y):
                    for x2 in range(0, SIZE_X):
                        for y2 in range(0, SIZE_Y):
                            q_table[((x1, y1), (x2, y2))] = np.ones(
                                NUMBER_OF_ACTIONS)  # [np.random.uniform(-5, 0) for i in range(NUMBER_OF_ACTIONS)]
        else:  # if we have a saved Q-table
            with open(start_q_table, "rb") as f:
                q_table = pickle.load(f)

        return q_table

    def update_context(self, new_state: State, reward, is_terminal):
        if self.UPDATE_CONTEXT:
            state_entry = (new_state.my_pos.get_tuple(), new_state.enemy_pos.get_tuple())
            action_entry = int(self._action)

            max_future_q = np.max(self._Q_matrix[state_entry])  # max Q value for this new observation

            current_q = self._Q_matrix[self._previous_state][action_entry]  # current Q for our chosen action

            if is_terminal:
                new_q = reward
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            self._Q_matrix[self._previous_state][action_entry] = new_q

            self._previous_state = state_entry
        self.update_epsilon()


    def update_epsilon(self):
        if self.IS_TRAINING:
            self._epsilon = max([self._epsilon * EPSILONE_DECAY, min_epsilon])  # change epsilon
        else:
            self._epsilon = 0

    def get_action(self, state: State)-> AgentAction:

        state_entry = (state.my_pos.get_tuple(), state.enemy_pos.get_tuple())

        if np.random.random() > self._epsilon or self.UPDATE_CONTEXT == False:
            # get the action
            # TODO:  check with Inbal
            action = np.argmax(self._Q_matrix[state_entry])
        else:
            action = np.random.randint(0, NUMBER_OF_ACTIONS)

        self._action = AgentAction(action)

        return self._action

    def type(self) -> AgentType:
        return self._type

    def get_epsolon(self):
        return self._epsilon

    def save_model(self, episodes_rewards, save_folder_path, color):
        number_of_rounds = len(episodes_rewards)-1
        if self._Q_matrix != None:
            if color==Color.Blue:
                with open(os.path.join(save_folder_path, f"qtable_blue-{number_of_rounds}.pickle"), "wb") as fb:
                    pickle.dump(self._Q_matrix, fb)
            if color==Color.Red:
                with open(os.path.join(save_folder_path, f"qtable_red-{number_of_rounds}.pickle"), "wb") as fb:
                    pickle.dump(self._Q_matrix, fb)




