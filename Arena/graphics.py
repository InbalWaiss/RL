import matplotlib.pyplot as plt
import numpy as np
from Arena.constants import *
from Arena.helper_funcs import *
from Arena.geometry import LOS, bresenham
import cv2


def print_stats(array_of_results, save_folder_path, plot_every, save_figure=True, steps=False):
    moving_avg = np.convolve(array_of_results, np.ones((plot_every,)) / plot_every, mode='valid')
    plt.figure()
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.xlabel("episode #")
    if steps:
        # plt.axis([0, len(array_of_results), 0, MAX_STEPS_PER_EPISODE])
        plt.axis([0, len(array_of_results), 0, max(moving_avg)])
        plt.suptitle(f"Avg number of steps per episode")
        plt.ylabel(f"steps per episode {SHOW_EVERY}ma")
        if save_figure:
            plt.savefig(save_folder_path + os.path.sep + '#steps_'+str(len(array_of_results)-1))
    else:
        plt.axis([0, len(array_of_results), -WIN_REWARD-50, WIN_REWARD+50])
        plt.suptitle(f"Rewards per episode")
        plt.ylabel(f"Reward {SHOW_EVERY}ma")
        if save_figure:
            plt.savefig(save_folder_path + os.path.sep + 'rewards_' + str(len(array_of_results)-1))

    plt.close()
    # plt.show()

def print_stats_humna_player(array_of_results, save_folder_path, number_of_episodes, save_figure=True, steps=False, red_player=False):
    moving_avg = np.convolve(array_of_results, np.ones((1,)) / 1, mode='valid')
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.xlabel("episode #")
    if steps:
        plt.axis([0, len(array_of_results), 0, MAX_STEPS_PER_EPISODE])
        if red_player: #number of steps figure for red player
            plt.suptitle(f"avg number of steps per episode red player")
            plt.ylabel(f"steps{number_of_episodes}")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + '#steps red player')
        else: #number of steps figure for blue player
            plt.suptitle(f"avg number of steps per episode blue player")
            plt.ylabel(f"steps{number_of_episodes}")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + '#steps blue player')
    else:
        plt.axis([0, len(array_of_results), -WIN_REWARD, WIN_REWARD])
        if red_player: #reward figure for red player
            plt.suptitle(f"Reward for red player")
            plt.ylabel(f"Rewards")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + 'rewards red player')
        else: #reward figure for blue player
            plt.suptitle(f"Reward for blue player")
            plt.ylabel(f"Rewards")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + 'rewards blue player')

    plt.show()

def print_episode_graphics(env, episode, last_step_number):
    blue = env.blue_player
    red = env.red_player
    game_number = episode.episode_number
    is_terminal = episode.is_terminal
    number_of_steps = last_step_number
    wins_for_blue = env.wins_for_blue
    wins_for_red = env.wins_for_red
    tie_count = env.tie_count

    # img = img.resize((600, 600))

    const = 30
    margin_x = 2
    margin_y = 0

    informative_env = np.zeros((const * (SIZE_X + margin_x * 2), const * (SIZE_X + margin_y * 2 ), 3), dtype=np.uint8)

    only_env = np.zeros((const * SIZE_X , const * SIZE_X , 3), dtype=np.uint8)
    # informative_env[env.blue_player.x][env.blue_player.y] = dict_of_colors[RED_N]
    # image[env.red_player.x][env.red_player.y] = dict_of_colors[BLUE_N]
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            if DSM[x][y] == 1.:
                only_env[x * const : x * const + const, y * const : + y * const + const] = dict_of_colors[GREY_N]


    # add margins to print information on
    informative_env[0:margin_x * const] = dict_of_colors[GREY_N]
    informative_env[(margin_x + SIZE_X) * const:(2 * margin_x + SIZE_X) * const] = dict_of_colors[GREY_N]
    informative_env[margin_x * const : (margin_x + SIZE_X) * const, margin_y * const : (margin_y +SIZE_Y) *const ] = only_env

    # # remove the blue tile
    # informative_env[(blue.x + margin_x) * const : (blue.x+margin_x) * const + const, (margin_y + blue.y) * const : (margin_y + blue.y) * const + const] = dict_of_colors[BLACK_N] #set tile black
    #
    # # remove the red tile
    # informative_env[(red.x + margin_x) * const : (red.x+margin_x) * const + const, (margin_y + red.y) * const : (margin_y + red.y) * const + const] = dict_of_colors[BLACK_N] #set tile black

    points_in_LOS = []
    # paint the tiles in line from blue to red in yellow
    _, points_in_LOS = check_if_LOS(blue.x, blue.y, red.x, red.y)
    for point in points_in_LOS:
        informative_env[(point[0] + margin_x) * const : (point[0] + margin_x) * const + const,
        (point[1] + margin_y) * const: (point[1]+margin_y) * const + const] = dict_of_colors[YELLOW_N]

    # set the players as circles
    radius = int(np.ceil(const/2))
    thickness = -1
    # set the red player
    center_cord_red_x = (red.x + margin_x) * const + radius
    center_cord_red_y = (red.y + margin_y) * const + radius
    red_color = dict_of_colors[RED_N]
    cv2.circle(informative_env, (center_cord_red_y, center_cord_red_x), radius, red_color, thickness)
    # set the blue player
    center_cord_blue_x = (blue.x + margin_x) * const + radius
    center_cord_blue_y = (blue.y + margin_y) * const + radius
    blue_color = dict_of_colors[BLUE_N]
    cv2.circle(informative_env, (center_cord_blue_y, center_cord_blue_x), radius, blue_color, thickness)

    # add episode number at the bottom of the window
    font = cv2.FONT_HERSHEY_SIMPLEX
    botoomLeftCornerOfText = (5, (SIZE_Y + margin_x * 2) * const - 15)
    fontScale = 0.5
    color = (100, 200, 120) #greenish
    thickness = 1
    cv2.putText(informative_env, f"episode #{game_number}", botoomLeftCornerOfText, font, 0.7, color, thickness, cv2.LINE_AA)

    if is_terminal:
        # print the dominating point
        red_over_blue_flag, red_over_blue_point = is_dominating(red, blue)
        blue_over_red_flag, blue_over_red_point = is_dominating(blue, red)
        if red_over_blue_flag:
            informative_env[(red_over_blue_point[0] + margin_x) * const + 3 : -3 + (red_over_blue_point[0] + margin_x) * const + const,
            (red_over_blue_point[1] + margin_y) *const + 3 : -3 +(red_over_blue_point[1] + margin_y) * const + const] = dict_of_colors[DARK_RED_N]
        if blue_over_red_flag:
            informative_env[(blue_over_red_point[0] + margin_x) * const + 3 : -3 + (blue_over_red_point[0] + margin_x) * const + const,
            (blue_over_red_point[1] + margin_y) *const + 3 : -3 +(blue_over_red_point[1] + margin_y) * const + const] = dict_of_colors[DARK_BLUE_N]

        # print who won
        thickness = 2
        botoomLeftCornerOfText_steps = (int(np.floor(SIZE_Y/2)) * const - 79, 55)
        if red_over_blue_flag and blue_over_red_flag:
            botoomLeftCornerOfText = (int(np.floor(SIZE_Y / 2))* const - 38, 30)
            cv2.putText(informative_env, f"TIE!", botoomLeftCornerOfText, font, fontScale, dict_of_colors[PURPLE_N], thickness, cv2.LINE_AA)
            cv2.putText(informative_env, f"after {number_of_steps} steps", botoomLeftCornerOfText_steps, font, 0.7, dict_of_colors[PURPLE_N], 0, cv2.LINE_AA)
        elif red_over_blue_flag:
            botoomLeftCornerOfText = (int(np.floor(SIZE_Y / 2))* const - 55, 30)
            cv2.putText(informative_env, f"RED WON!", botoomLeftCornerOfText, font, fontScale, dict_of_colors[RED_N], thickness-1, cv2.LINE_AA)
            cv2.putText(informative_env, f"after {number_of_steps} steps", botoomLeftCornerOfText_steps, font, 0.7, dict_of_colors[PURPLE_N], 0, cv2.LINE_AA)
        elif blue_over_red_flag:
            botoomLeftCornerOfText = (int(np.floor(SIZE_Y / 2))* const - 50, 30)
            cv2.putText(informative_env, f"BLUE WON!", botoomLeftCornerOfText, font, fontScale, dict_of_colors[BLUE_N], thickness-1, cv2.LINE_AA)
            cv2.putText(informative_env, f"after {number_of_steps} steps", botoomLeftCornerOfText_steps, font, 0.7, dict_of_colors[PURPLE_N], 0, cv2.LINE_AA)
        else: # both lost...
            botoomLeftCornerOfText = (int(np.floor(SIZE_Y / 2))* const - 60, 30)
            cv2.putText(informative_env, f"both lost...", botoomLeftCornerOfText, font, fontScale, dict_of_colors[PURPLE_N], thickness-1, cv2.LINE_AA)
            cv2.putText(informative_env, f"after {number_of_steps} steps", botoomLeftCornerOfText_steps, font, 0.7, dict_of_colors[PURPLE_N], 0, cv2.LINE_AA)

    else: # not terminal state
        botoomLeftCornerOfText = (int(np.floor(SIZE_Y / 2))* const - 45, 20)
        cv2.putText(informative_env, f"steps: {number_of_steps}", botoomLeftCornerOfText, font, fontScale, dict_of_colors[PURPLE_N], 0, cv2.LINE_AA)

    # print number of wins
    botoomLeftCornerOfText = (5, 15)
    cv2.putText(informative_env, f"Blue wins: {wins_for_blue}", botoomLeftCornerOfText, font, fontScale, dict_of_colors[BLUE_N], 0,
                cv2.LINE_AA)
    botoomLeftCornerOfText = (5, 35)
    cv2.putText(informative_env, f"Red wins: {wins_for_red}", botoomLeftCornerOfText, font, fontScale, dict_of_colors[RED_N], 0,
                cv2.LINE_AA)
    botoomLeftCornerOfText = (5, 55)
    cv2.putText(informative_env, f"Tie : {tie_count}", botoomLeftCornerOfText, font, fontScale, dict_of_colors[PURPLE_N], 0,
                cv2.LINE_AA)

    cv2.imshow("informative_env", np.array(informative_env)) #show it!
    cv2.waitKey(1)


def print_episode_graphics_old(env, episode, last_step_number):
    #     blue, red, game_number, is_terminal, number_of_steps, wins_for_blue, wins_for_red, tie_count):
    # env, last_step_number

    blue = env.blue_player
    red = env.red_player
    game_number = episode.episode_number
    is_terminal = episode.is_terminal
    number_of_steps = last_step_number
    wins_for_blue = env.wins_for_blue
    wins_for_red = env.wins_for_red
    tie_count = env.tie_count

    const = 50
    env = np.zeros((SIZE_X * const, SIZE_Y * const, 3), dtype=np.uint8)  # starts an RBG of our size

    # paint the tiles in line from blue to red in yellow
    if PRINT_TILES_IN_LOS:
        _, points_in_LOS = check_if_LOS(blue.x, blue.y, red.x, red.y)
        for point in points_in_LOS:
            env[point[0] * const:point[0] * const + const, point[1] * const:point[1] * const + const] = dict_of_colors[
                YELLOW_N]  # sets all the tiles in LOS green

    # paint the blue_player and the red_player
    radius = int(np.ceil(const / 2))
    thickness = -1

    center_cor_blue_x = blue.x * const + radius
    center_cor_blue_y = blue.y * const + radius
    blue_color = dict_of_colors[BLUE_N]
    cv2.circle(env, (center_cor_blue_y, center_cor_blue_x), radius, blue_color, thickness)

    center_cor_red_x = red.x * const + radius
    center_cor_red_y = red.y * const + radius
    red_color = dict_of_colors[RED_N]
    cv2.circle(env, (center_cor_red_y, center_cor_red_x), radius, red_color, thickness)

    # print domination points
    red_over_blue_flag, red_over_blue_point = is_dominating(red, blue)
    blue_over_red_flag, blue_over_red_point = is_dominating(blue, red)
    if is_terminal:
        if red_over_blue_flag:
            env[red_over_blue_point[0] * const:red_over_blue_point[0] * const + const,
            red_over_blue_point[1] * const:red_over_blue_point[1] * const + const] = dict_of_colors[
                DARK_RED_N]  # sets the green location tile to green color
        if blue_over_red_flag:
            env[blue_over_red_point[0] * const:blue_over_red_point[0] * const + const,
            blue_over_red_point[1] * const:blue_over_red_point[1] * const + const] = dict_of_colors[
                DARK_BLUE_N]  # sets the green location tile to green color

    # set the obstacle tile to grey
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            if DSM[x][y] == 1.:
                env[x * const:x * const + const, y * const:y * const + const] = dict_of_colors[GREY_N]

    cv2.waitKey(35)  # delay of refresh
    font = cv2.FONT_HERSHEY_SIMPLEX
    bootomLeftCornerOfText = (5, SIZE_Y * const - 15)
    fontScale = 1
    color = (100, 200, 120)  # greenish
    thickness = 2
    cv2.putText(env, f"episode #{game_number}", bootomLeftCornerOfText, font, fontScale, color, thickness,
                cv2.LINE_AA)

    # print who won
    if is_terminal or number_of_steps==MAX_STEPS_PER_EPISODE:
        thickness = 3
        bootomLeftCornerOfText_steps = (int(np.floor(SIZE_Y / 2)) * const - 79, 55)
        if red_over_blue_flag and blue_over_red_flag:
            bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const - 38, 30)
            cv2.putText(env, f"TIE!", bootomLeftCornerOfText, font, fontScale, dict_of_colors[
                PURPLE_N], thickness,
                        cv2.LINE_AA)
            cv2.putText(env, f"after {number_of_steps} steps", bootomLeftCornerOfText_steps, font, 0.7, dict_of_colors[
                PURPLE_N], 0,
                        cv2.LINE_AA)
        elif red_over_blue_flag:
            bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const - 80, 30)
            cv2.putText(env, f"RED WON!", bootomLeftCornerOfText, font, fontScale, dict_of_colors[
                RED_N], thickness,
                        cv2.LINE_AA)
            cv2.putText(env, f"after {number_of_steps} steps", bootomLeftCornerOfText_steps, font, 0.7, dict_of_colors[
                PURPLE_N], 0,
                        cv2.LINE_AA)
        elif blue_over_red_flag:
            bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const - 90, 30)
            cv2.putText(env, f"BLUE WON!", bootomLeftCornerOfText, font, fontScale, dict_of_colors[
                BLUE_N], thickness,
                        cv2.LINE_AA)
            cv2.putText(env, f"after {number_of_steps} steps", bootomLeftCornerOfText_steps, font, 0.7, dict_of_colors[
                PURPLE_N], 0,
                        cv2.LINE_AA)
        else: #both lost...
            bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const - 90, 30)
            cv2.putText(env, f"both lost...", bootomLeftCornerOfText, font, fontScale, dict_of_colors[
                PURPLE_N], thickness-1,
                        cv2.LINE_AA)
            cv2.putText(env, f"after {number_of_steps} steps", bootomLeftCornerOfText_steps, font, 0.7, dict_of_colors[
                PURPLE_N], 0,
                        cv2.LINE_AA)

    else: #not terminal state
        bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const -45, 20)
        cv2.putText(env, f"steps: {number_of_steps}", bootomLeftCornerOfText, font, 0.7, dict_of_colors[
            PURPLE_N], 0,
                    cv2.LINE_AA)

    #print number of wins
    bootomLeftCornerOfText = (5, 20)
    cv2.putText(env, f"Blue won: {wins_for_blue}", bootomLeftCornerOfText, font, 0.7, dict_of_colors[
        BLUE_N], 0,
                cv2.LINE_AA)
    bootomLeftCornerOfText = (5, 40)
    cv2.putText(env, f"Red won: {wins_for_red}", bootomLeftCornerOfText, font, 0.7, dict_of_colors[
        RED_N], 0,
                cv2.LINE_AA)
    bootomLeftCornerOfText = (5, 60)
    cv2.putText(env, f"Tie: {tie_count}", bootomLeftCornerOfText, font, 0.7, dict_of_colors[
        PURPLE_N], 0,
                cv2.LINE_AA)


    cv2.imshow(f"state as ((blue_cor), (red_cor))", env)
    if is_terminal:  # if we reached the end of the episode
        if cv2.waitKey(500) & 0xFF == ord('q'):
            cv2.destroyWindow(f"state as ((blue_cor), (red_cor))")
            time.sleep(2)
            pass
        time.sleep(2)
    else:
        if cv2.waitKey(3) & 0xFF == ord('q'):
            cv2.destroyWindow(f"state as ((blue_cor), (red_cor))")
            pass

