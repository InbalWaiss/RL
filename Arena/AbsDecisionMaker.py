import abc
from typing import Tuple

from Arena.CState import State
from Arena.constants import AgentAction


class AbsDecisionMaker(metaclass=abc.ABCMeta):

    def update_context(self, new_state: State, reward: Tuple[int, int], is_terminal):

        pass

    def get_action(self, state: State)-> AgentAction:

        pass

    def set_state(self, state: State):

        pass