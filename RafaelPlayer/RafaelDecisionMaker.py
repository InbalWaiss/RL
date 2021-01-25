from Arena.CState import State
from Arena.AbsDecisionMaker import AbsDecisionMaker
from RafaelPlayer.QPlayer_constants import *
from Arena.constants import *
import numpy as np
import os
import pickle


class RafaelDecisionMaker(AbsDecisionMaker):

    def __init__(self, agent_str=None):

        self._previous_state = {}
        self._action = -1
        self._epsilon = epsilon
        self._type = AgentType.Q_table
        self.episode_number = 0


        if agent_str is not None:
            p = path.join(RELATIVE_PATH_HUMAN_VS_MACHINE_DATA, agent_str)
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

        self._epsilon = max([self._epsilon * EPSILONE_DECAY, 0.05])  # change epsilon

    def get_action(self, state: State)-> AgentAction:

        state_entry = (state.my_pos.get_tuple(), state.enemy_pos.get_tuple())

        # print(obs)
        if np.random.random() > self._epsilon:
            # get the action
            # TODO:  check with Inbal
            action = np.argmax(self._Q_matrix[state_entry])
        else:
            action = np.random.randint(0, NUMBER_OF_ACTIONS)

        self._action = AgentAction(action)

        return self._action

    def type(self) -> AgentType:
        return self._type

    def save_model(self, number_of_rounds, save_folder_path, color):
        if self._Q_matrix != None:
            if color==Color.Blue:
                with open(os.path.join(save_folder_path, f"qtable_blue-{number_of_rounds}.pickle"), "wb") as fb:
                    pickle.dump(self._Q_matrix, fb)
            if color==Color.Red:
                with open(os.path.join(save_folder_path, f"qtable_red-{number_of_rounds}.pickle"), "wb") as fb:
                    pickle.dump(self._Q_matrix, fb)




