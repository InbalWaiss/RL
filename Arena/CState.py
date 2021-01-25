from Arena.Position import Position
import numpy as np
from constants import *
from PIL import Image


class State(object):

    def __init__(self, my_pos: Position, enemy_pos: Position):

        self.my_pos = my_pos
        self.enemy_pos = enemy_pos
        self.img = self.get_image()



    def get_image(self):
        env = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8) # starts an rbg of small world
        env[self.my_pos._x][self.my_pos._y] = dict_of_colors[BLUE_N]
        env[self.enemy_pos._x][self.enemy_pos._y] = dict_of_colors[RED_N]
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x][y] == 1.:
                    env[x][y] = dict_of_colors[GREY_N]
        # img = Image.fromarray(env).convert('L').resize((SIZE_X, SIZE_Y), Image.BILINEAR)
        img = Image.fromarray(env, 'RGB')
        # Image._show(img)
        return env

