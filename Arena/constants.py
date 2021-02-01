from enum import IntEnum

import numpy as np
from os import path


WITH_LOS = True
PRINT_TILES_IN_LOS = True
USE_BRESENHAM_LINE = False


SIZE_X = 15
SIZE_Y = 15

MOVE_PENALTY = 1
WIN_REWARD = 250 #will be change to be reward for reaching controling point
LOST_PENALTY = 250
TIE = 0

MAX_STEPS_PER_EPISODE = 250
NUMBER_OF_ACTIONS = 9

BLUE_N = 1 #blue player key in dict
DARK_BLUE_N = 2
RED_N = 3 #red player key in dict
DARK_RED_N = 4
PURPLE_N = 5
YELLOW_N = 6 #to be used for line from blue to red
GREY_N = 7 #obstacle key in dict
GREEN_N = 8
BLACK_N = 9

dict_of_colors = {1: (255, 175, 0),  # blueish color
                  2: (150, 100, 0), #darker blue
                  3: (0, 0, 255), # red
                  4: (0, 0, 150), #dark red
                  5: (230, 100, 150), #purple
                  6: (60, 255, 255), #yellow
                  7: (100, 100, 100),#grey
                  8: (0, 255, 0),#green
                  9: (0, 0, 0) #black
                  }



OBSTACLE = 1.

#1 is an obstacle
DSM = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
])



#save information
COMMON_PATH = path.dirname(path.realpath(__file__))
MAIN_PATH = path.dirname(COMMON_PATH)
OUTPUT_DIR = path.join(MAIN_PATH, 'Arena')
STATS_RESULTS_RELATIVE_PATH = path.join(OUTPUT_DIR, 'statistics')
RELATIVE_PATH_HUMAN_VS_MACHINE_DATA = path.join(MAIN_PATH, 'RafaelPlayer/trained_agents')

EASY_AGENT = 'easy.pickle'
MEDIUM_AGENT = 'medium.pickle'
HARD_AGENT = 'hard.pickle'

SHOW_EVERY = 500
NUM_OF_EPISODES = 1_000_000
SAVE_STATS_EVERY = 50_000


class AgentAction(IntEnum):

    TopRight = 1
    Right = 2
    BottomRight = 3
    Bottom = 4
    Stay = 5
    Top = 6
    BottomLeft = 7
    Left = 8
    TopLeft = 0

class AgentType(IntEnum):
    Q_table = 1
    DQN_basic = 2
    DQN_keras = 3
    DQN_temporalAttention = 4

Agent_type_str = {AgentType.Q_table : "Q_table",
                  AgentType.DQN_basic : "DQN",
                  AgentType.DQN_keras : "DQN_keras",
                  AgentType.DQN_temporalAttention : "DQN_temporalAttention"}

class Color(IntEnum):
    Blue = 1
    Red = 2


