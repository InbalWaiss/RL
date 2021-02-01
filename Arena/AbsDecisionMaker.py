import abc
from typing import Tuple

from Arena.CState import State
from Arena.constants import AgentAction, AgentType


class AbsDecisionMaker(metaclass=abc.ABCMeta):

    def update_context(self, new_state: State, reward, is_terminal):

        pass

    def get_action(self, state: State)-> AgentAction:

        pass

    def set_initial_state(self, state: State):

        pass

    def type(self)-> AgentType:

        pass

    def get_epsolon(self):

        return -1

    def save_model(self, number_of_rounds):

        pass
