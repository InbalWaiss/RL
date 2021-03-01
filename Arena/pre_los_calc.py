
from Arena.constants import *
from PIL import Image, ImageFilter
import pickle
import numpy as np
from Arena.helper_funcs import check_if_LOS
import matplotlib.pyplot as plt
from Arena.Environment import Environment, Episode
from Arena.Entity import Entity

def creat_and_save_dictionaries():
    los_from_pos = {}
    no_los_from_pos = {}
    for x1 in range(0, SIZE_X):
        for y1 in range(0, SIZE_Y):
            los_from_pos[(x1, y1)] = []
            no_los_from_pos[(x1, y1)] = []
            for x2 in range(0, SIZE_X):
                for y2 in range(0, SIZE_Y):
                    is_los, _ = check_if_LOS(x1, y1, x2, y2)
                    if is_los:
                        los_from_pos[(x1, y1)].append((x2, y2))
                    else:
                        if DSM[x2, y2]!=1:
                            no_los_from_pos[(x1, y1)].append((x2, y2))



    save_obj(los_from_pos, "dictionary_position_los")
    save_obj(no_los_from_pos, "dictionary_position_no_los")


def show_LOS_from_point(x1,y1):
    env = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8)  # starts an rbg of small world

    points_in_LOS = DICT_POS_LOS[(x1,y1)]
    for point in points_in_LOS:
        env[point[0]][point[1]] = (100, 0, 0)

    env[x1][y1] = dict_of_colors[BLUE_N]

    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            if DSM[x][y] == 1.:
                env[x][y] = dict_of_colors[GREY_N]

    plt.matshow(env)
    plt.show()

def show_no_LOS_from_point(x1,y1):
    env = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8)  # starts an rbg of small world

    points_in_LOS = DICT_POS_NO_LOS[(x1,y1)]
    for point in points_in_LOS:
        env[point[0]][point[1]] = (100, 0, 0)

    env[x1][y1] = dict_of_colors[BLUE_N]

    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            if DSM[x][y] == 1.:
                env[x][y] = dict_of_colors[GREY_N]

    plt.matshow(env)
    plt.show()

def find_closest_point_not_in_los(x1,y1):
    arr = np.asarray(DICT_POS_NO_LOS[(x1, y1)])
    value = np.array([x1, y1])
    closest_point_no_loss = arr[np.linalg.norm(arr - value, axis=1).argmin()]
    env = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8)  # starts an rbg of small world
    points_in_LOS = DICT_POS_LOS[(x1, y1)]
    for point in points_in_LOS:
        env[point[0]][point[1]] = (100, 0, 0)  # dict_of_colors[GREEN_N]
    env[x1][y1] = dict_of_colors[BLUE_N]
    env[closest_point_no_loss[0]][closest_point_no_loss[1]] = (200, 200, 200)
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            if DSM[x][y] == 1.:
                env[x][y] = dict_of_colors[GREY_N]
    plt.matshow(env)
    plt.show()
    return closest_point_no_loss

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def calc_and_save_dominating_points():
    dominating_points_dict = {}
    for x1 in range(0, SIZE_X):
        for y1 in range(0, SIZE_Y):
            goal_points = find_dominating_point(x1, y1)
            dominating_points_dict[(x1,y1)] = goal_points

    save_obj(dominating_points_dict, "dictionary_dominating_points")


def find_dominating_point(x1, y1):
    point1 = (x1, y1)
    arr = DICT_POS_LOS[(x1, y1)]
    goal_points = []
    for p in arr:
        if can_escape_by_one_step(point1, p):
            goal_points.append(p)

    if True:
        img_env = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8)  # starts an rbg of small world
        points_in_LOS = DICT_POS_LOS[(x1, y1)]
        for point in points_in_LOS:
            img_env[point[0]][point[1]] = (100, 0, 0)  # dict_of_colors[GREEN_N]
        img_env[x1][y1] = dict_of_colors[BLUE_N]

        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x][y] == 1.:
                    img_env[x][y] = dict_of_colors[GREY_N]
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img_env)

        for point in goal_points:
            img_env[point[0]][point[1]] = dict_of_colors[GREEN_N]

        axs[1].imshow(img_env)
        plt.show()
    print("end ", x1, y1)
    return goal_points

def can_escape_by_one_step(point1, point2):
    env = Environment()

    env.red_player = Entity()
    env.red_player.x = point1[0]
    env.red_player.y = point1[1]

    env.blue_player = Entity()
    env.blue_player.x = point2[0]
    env.blue_player.y = point2[1]

    win_stat = env.compute_terminal()

    if win_stat==WinEnum.Blue:
        return True

    return False


if __name__ == '__main__':
    show_LOS_from_point(5, 5)
    find_closest_point_not_in_los(5, 5)
    find_dominating_point(5, 5)