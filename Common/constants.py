from enum import IntEnum

import numpy as np
from os import path
import pickle

PRINT_TILES_IN_LOS = False
USE_BRESENHAM_LINE = False
USE_LOS_IN_STATE = False

DANGER_ZONE_IN_STATE = True
DOMINATING_POINTS_IN_STATE = False
LOSE_POINTS_IN_STATE = False
ACTION_SPACE_9 = True
ACTION_SPACE_4 = False
if not ACTION_SPACE_9:
    ACTION_SPACE_4 = True

RED_PLAYER_MOVES = True
FIXED_START_POINT_RED = True
FIXED_START_POINT_BLUE = True

FIRE_RANGE_FLAG = True
FIRE_RANGE = 7

ZERO_SUM_GAME = False
CLIP_REWARD_1 = False

#image state mode
IMG_STATE_MODE = 'L'
#IMG_STATE_MODE= 'P'

STR_FOLDER_NAME = ""

try:
    with open('Common/Preprocessing/dictionary_position_los.pkl', 'rb') as f:
        DICT_POS_LOS = pickle.load(f)
except:
    try:
        with open('dictionary_position_los.pkl', 'rb') as f:
            DICT_POS_LOS = pickle.load(f)
    except:
        try:
            with open('../../../../../../קוד עם באג רשת מתכנסת ל-1/Common/Preprocessing/dictionary_position_los.pkl', 'rb') as f:
                DICT_POS_LOS = pickle.load(f)
        except:
            pass

try:
    with open('Common/Preprocessing/dictionary_position_no_los.pkl', 'rb') as f:
        DICT_POS_NO_LOS = pickle.load(f)
except:
    try:
        with open('dictionary_position_no_los.pkl', 'rb') as f:
            DICT_POS_NO_LOS = pickle.load(f)
    except:
        try:
            with open('../../../../../../קוד עם באג רשת מתכנסת ל-1/Common/Preprocessing/dictionary_position_no_los.pkl', 'rb') as f:
                DICT_POS_NO_LOS = pickle.load(f)
        except:
            pass

try:
    with open('Common/Preprocessing/dictionary_dominating_points.pkl', 'rb') as f:
        DICT_DOMINATING_POINTS = pickle.load(f)
except:
    try:
        with open('dictionary_dominating_points.pkl', 'rb') as f:
            DICT_DOMINATING_POINTS = pickle.load(f)
    except:
        try:
            with open(
                    '../../../../../../קוד עם באג רשת מתכנסת ל-1/Common/Preprocessing/dictionary_dominating_points.pkl', 'rb') as f:
                DICT_DOMINATING_POINTS = pickle.load(f)
        except:
            pass

try:
    with open('Common/Preprocessing/dictionary_lose_points.pkl', 'rb') as f:
        DICT_LOSE_POINTS = pickle.load(f)
except:
    try:
        with open('dictionary_lose_points.pkl', 'rb') as f:
            DICT_LOSE_POINTS = pickle.load(f)
    except:
        try:
            with open('../../../../../../קוד עם באג רשת מתכנסת ל-1/Common/Preprocessing/dictionary_lose_points.pkl', 'rb') as f:
                DICT_LOSE_POINTS = pickle.load(f)
        except:
            pass

SIZE_X = 15
SIZE_Y = 15

MOVE_PENALTY = 0.1
WIN_REWARD = 20 #will be change to be reward for reaching controling point
LOST_PENALTY = -WIN_REWARD
TIE = 0

# MOVE_PENALTY = 0.1
# WIN_REWARD = 1 #will be change to be reward for reaching controling point
# LOST_PENALTY = -WIN_REWARD
# TIE = 0

# MOVE_PENALTY = 5
# WIN_REWARD = 120 #will be change to be reward for reaching controling point
# LOST_PENALTY = -WIN_REWARD
# TIE = 0


MAX_STEPS_PER_EPISODE = 200
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
BRIGHT_RED = 10
BRIGHT_BRIGHT_RED = 11

class WinEnum(IntEnum):

    Blue = 0
    Red = 1
    Tie = 2
    NoWin = 3
    #Done = 4


# dict_of_colors = {1: (255, 0, 0),  #blue
#                   2: (230, 0, 0), #darker blue
#                   3: (0, 0, 255), # red
#                   4: (0, 0, 230), #dark red
#                   5: (230, 100, 150), #purple
#                   6: (60, 255, 255), #yellow
#                   7: (100, 100, 100),#grey
#                   8: (0, 255, 0),#green
#                   9: (0, 0, 0), #black
#                   10: (0, 0, 75), #bright red
#                   11: (0, 0, 25) #bright bright red
#                   }

dict_of_colors = {1: (255, 0, 0),  #blue
                  2: (175, 0, 0), #darker blue
                  3: (0, 0, 255), # red
                  4: (0, 0, 175), #dark red
                  5: (230, 100, 150), #purple
                  6: (60, 255, 255), #yellow
                  7: (100, 100, 100),#grey
                  8: (0, 255, 0),#green
                  9: (0, 0, 0), #black
                  10: (0, 0, 75), #bright red
                  11: (0, 0, 25) #bright bright red
                  }

# dict_of_colors = {1: (255, 0, 0),  #blue
#                   2: (150, 0, 0), #darker blue
#                   3: (0, 0, 255), # red
#                   4: (0, 0, 150), #dark red
#                   5: (230, 100, 150), #purple
#                   6: (60, 255, 255), #yellow
#                   7: (100, 100, 100),#grey
#                   8: (0, 255, 0),#green
#                   9: (0, 0, 0), #black
#                   10: (0, 0, 75), #bright red
#                   11: (0, 0, 25) #bright bright red
#                   }



OBSTACLE = 1.

#1 is an obstacle
DSM = np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
])



DSM_30X30 =  np.array([
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
])


if ACTION_SPACE_9:
    NUMBER_OF_ACTIONS = 9
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


else: # ACTION_SPACE = 4
    NUMBER_OF_ACTIONS = 4
    class AgentAction(IntEnum):
        Right = 0
        Bottom = 1
        Top = 2
        Left = 3


class AgentType(IntEnum):
    Q_table = 1
    DQN_basic = 2
    DQN_keras = 3
    DQN_temporalAttention = 4
    DQNAgent_spatioalAttention = 5
    Greedy = 6

Agent_type_str = {AgentType.Q_table : "Q_table",
                  AgentType.DQN_basic : "DQN_basic",
                  AgentType.DQN_keras : "DQN_keras",
                  AgentType.DQN_temporalAttention : "DQN_temporalAttention",
                  AgentType.DQNAgent_spatioalAttention : "DQNAgent_spatioalAttention",
                  AgentType.Greedy : "Greedy_player"}

class Color(IntEnum):
    Blue = 1
    Red = 2

#save information
COMMON_PATH = path.dirname(path.realpath(__file__))
MAIN_PATH = path.dirname(COMMON_PATH)
OUTPUT_DIR = path.join(MAIN_PATH, 'Arena')
STATS_RESULTS_RELATIVE_PATH = path.join(OUTPUT_DIR, '../Arena/statistics')
RELATIVE_PATH_HUMAN_VS_MACHINE_DATA = path.join(MAIN_PATH, 'Qtable/trained_agents')


USE_DISPLAY = True
SHOW_EVERY = 1000
NUM_OF_EPISODES = 1_000_000
SAVE_STATS_EVERY = 50000

# params to evaluate trained models
EVALUATE_SHOW_EVERY = 1
EVALUATE_NUM_OF_EPISODES = 100
EVALUATE_SAVE_STATS_EVERY = 100

EVALUATE_PLAYERS_EVERY = 1000

# training mode
IS_TRAINING = False
UPDATE_RED_CONTEXT = True
UPDATE_BLUE_CONTEXT = True

if not IS_TRAINING:
    UPDATE_RED_CONTEXT=False
    UPDATE_BLUE_CONTEXT=False
