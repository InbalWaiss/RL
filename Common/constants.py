from enum import IntEnum

import numpy as np
from os import path
import pickle
from Common.Preprocessing.load_DSM_from_excel import get_DSM_berlin, get_DSM_Boston, get_DSM_Paris

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
FIXED_START_POINT_RED = False
FIXED_START_POINT_BLUE = False

FIRE_RANGE_FLAG = True
FIRE_RANGE = 7

#image state mode
IMG_STATE_MODE = 'L'
#IMG_STATE_MODE= 'P'

FULLY_CONNECTED = False
STR_FOLDER_NAME = "conv_Berlin"

#1 is an obstacle
DSM_names = {"15X15", "100X100_Berlin", "100X100_Paris", "100X100_Boston"}
DSM_name = "100X100_Berlin"
if DSM_name=="15X15":
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
    SIZE_X = 15
    SIZE_Y = 15
    MAX_STEPS_PER_EPISODE = 100

elif DSM_name=="100X100_Berlin":
    DSM = get_DSM_berlin()
    SIZE_X=100
    SIZE_Y=100
    FIRE_RANGE = 15
    MAX_STEPS_PER_EPISODE = 250

elif DSM_name=="100X100_Paris":
    DSM = get_DSM_Paris()
    SIZE_X=100
    SIZE_Y=100
    FIRE_RANGE = 15
    MAX_STEPS_PER_EPISODE = 250

elif DSM_name=="100X100_Boston":
    DSM = get_DSM_Boston()
    SIZE_X=100
    SIZE_Y=100
    FIRE_RANGE = 15
    MAX_STEPS_PER_EPISODE = 250
# if False:
#     import matplotlib.pyplot as plt
#     plt.matshow(DSM)
#     plt.show()


try:
    with open('Common/Preprocessing/dictionary_position_los_'+DSM_name+'.pkl', 'rb') as f:
        DICT_POS_LOS = pickle.load(f)
except:
    try:
        with open('dictionary_position_los_'+DSM_name+'.pkl', 'rb') as f:
            DICT_POS_LOS = pickle.load(f)
    except:
        try:
            with open('../Common/Preprocessing/dictionary_position_los_'+DSM_name+'.pkl', 'rb') as f:
                DICT_POS_LOS = pickle.load(f)
        except:
            pass

try:
    with open('Common/Preprocessing/dictionary_position_no_los_' + DSM_name + '.pkl', 'rb') as f:
        DICT_POS_NO_LOS = pickle.load(f)
except:
    try:
        with open('dictionary_position_no_los_' +DSM_name+ '.pkl', 'rb') as f:
            DICT_POS_NO_LOS = pickle.load(f)
    except:
        try:
            with open('../Common/Preprocessing/dictionary_position_no_los_' +DSM_name+ '.pkl', 'rb') as f:
                DICT_POS_NO_LOS = pickle.load(f)
        except:
            pass

try:
    with open('Common/Preprocessing/dictionary_dominating_points_'+DSM_name+'.pkl', 'rb') as f:
        DICT_DOMINATING_POINTS = pickle.load(f)
except:
    try:
        with open('dictionary_dominating_points_'+DSM_name+'.pkl', 'rb') as f:
            DICT_DOMINATING_POINTS = pickle.load(f)
    except:
        try:
            with open('../Common/Preprocessing/dictionary_dominating_points_'+DSM_name+'.pkl','rb') as f:
                DICT_DOMINATING_POINTS = pickle.load(f)
        except:
            pass

try:
    with open('Common/Preprocessing/dictionary_lose_points_'+DSM_name+'.pkl', 'rb') as f:
        DICT_LOSE_POINTS = pickle.load(f)
except:
    try:
        with open('dictionary_lose_points_'+DSM_name+'.pkl', 'rb') as f:
            DICT_LOSE_POINTS = pickle.load(f)
    except:
        try:
            with open('../Common/Preprocessing/dictionary_lose_points_'+DSM_name+'.pkl', 'rb') as f:
                DICT_LOSE_POINTS = pickle.load(f)
        except:
            pass



MOVE_PENALTY = 0.1
WIN_REWARD = 20
LOST_PENALTY = -1
TIE = 0


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

USE_OLD_COLORS = False
# for better separation of colors
dict_of_colors_for_state = {1: (0, 0, 255),  #blue
                  2: (0, 0, 175), #darker blue
                  3: (255, 0, 0), # red
                  4: (175, 0, 0), #dark red
                  5: (230, 100, 150), #purple
                  6: (60, 255, 255), #yellow
                  7: (100, 100, 100),#grey
                  8: (0, 255, 0),#green
                  9: (0, 0, 0), #black
                  10: (0, 0, 75), #bright red
                  11: (0, 0, 25) #bright bright red
                  }

dict_of_colors_for_graphics = {1: (255, 0, 0),  #blue
                               2: (175, 0, 0),  #darker blue
                               3: (0, 0, 255),  # red
                               4: (0, 0, 175),  #dark red
                               5: (230, 100, 150),  #purple
                               6: (60, 255, 255),  #yellow
                               7: (100, 100, 100),  #grey
                               8: (0, 255, 0),  #green
                               9: (0, 0, 0),  #black
                               10: (0, 0, 75),  #bright red
                               11: (0, 0, 25)  #bright bright red
                               }

OBSTACLE = 1.

if USE_OLD_COLORS:
    dict_of_colors_for_state = dict_of_colors_for_graphics





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
SHOW_EVERY =25
NUM_OF_EPISODES = 3_000_000
SAVE_STATS_EVERY = 2000

# params to evaluate trained models
EVALUATE_SHOW_EVERY = 1
EVALUATE_NUM_OF_EPISODES = 100
EVALUATE_SAVE_STATS_EVERY = 100

EVALUATE_PLAYERS_EVERY = 25

# training mode
IS_TRAINING = True
UPDATE_RED_CONTEXT = True
UPDATE_BLUE_CONTEXT = True

if not IS_TRAINING:
    UPDATE_RED_CONTEXT=False
    UPDATE_BLUE_CONTEXT=False
