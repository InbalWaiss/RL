from Arena.Position import Position
import numpy as np
from Arena.constants import *
from PIL import Image
from Arena.helper_funcs import check_if_LOS
import matplotlib.pyplot as plt


class State(object):

    def __init__(self, my_pos: Position, enemy_pos: Position):

        self.my_pos = my_pos
        self.enemy_pos = enemy_pos
        self.img = self.get_image()



    def get_image(self):
        env = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8) # starts an rbg of small world

        if USE_LOS_IN_STATE:
            _, points_in_LOS = check_if_LOS(self.my_pos._x, self.my_pos._y, self.enemy_pos._x, self.enemy_pos._y)
            for point in points_in_LOS:
                env[point[0]][point[1]]= dict_of_colors[GREEN_N]

        if DANGER_ZONE_IN_STATE:
            points_in_enemy_los = DICT_POS_LOS[(self.enemy_pos._x, self.enemy_pos._y)]
            for point in points_in_enemy_los:
                env[point[0]][point[1]] = dict_of_colors[DARK_RED_N]

        if DOMINATING_POINTS_IN_STATE:
            dominating_points = DICT_DOMINATING_POINTS[(self.enemy_pos._x, self.enemy_pos._y)]
            for point in dominating_points:
                env[point[0]][point[1]] = dict_of_colors[GREEN_N]

        if LOSE_POINTS_IN_STATE:
            lose_points = DICT_LOSE_POINTS[(self.enemy_pos._x, self.enemy_pos._y)]
            for point in lose_points:
                env[point[0]][point[1]] = dict_of_colors[DARK_RED_N]


        env[self.my_pos._x][self.my_pos._y] = dict_of_colors[BLUE_N]
        env[self.enemy_pos._x][self.enemy_pos._y] = dict_of_colors[RED_N]
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x][y] == 1.:
                    env[x][y] = dict_of_colors[GREY_N]
        if False:
            # print state for debug
            plt.matshow(env)
            plt.show()

        return env

