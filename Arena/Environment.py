
import time

from Qtable.Qtable_DecisionMaker import *
from Qtable.QPlayer_constants import START_EPSILON, EPSILONE_DECAY, LEARNING_RATE, DISCOUNT
from Arena.Position import Position
from Arena.graphics import print_stats, print_episode_graphics, save_win_statistics, save_reward_stats
from Arena.helper_funcs import *
from Common.constants import *

import numpy as np
from PIL import Image
import pandas as pd


class Environment(object):
    def __init__(self, TRAIN=True):
        self.episodes_rewards_blue = []
        self.episodes_rewards_blue.append(0)
        self.episodes_rewards_red = []
        self.episodes_rewards_red.append(0)
        self.win_array = []
        self.steps_per_episode = []
        self.steps_per_episode.append(0)
        self.number_of_steps = 0
        self.wins_for_blue = 0
        self.wins_for_red = 0
        self.tie_count = 0
        self.starts_at_win = 0
        self.starts_at_win_in_last_SHOW_EVERY_games = 0
        self.win_status: WinEnum = WinEnum.NoWin

        self.blue_epsilon_values = []
        self.blue_epsilon_values.append(1)

        self.blue_player = None
        self.red_player = None

        if TRAIN:
            self.SHOW_EVERY = SHOW_EVERY
            self.NUMBER_OF_EPISODES = NUM_OF_EPISODES

        else:
            self.SHOW_EVERY = EVALUATE_SHOW_EVERY
            self.NUMBER_OF_EPISODES = EVALUATE_NUM_OF_EPISODES

        self.create_path_for_statistics()

        self.end_game_flag = False

    def create_path_for_statistics(self):
        save_folder_path = path.join(STATS_RESULTS_RELATIVE_PATH,
                                     format(f"{str(time.strftime('%d'))}_{str(time.strftime('%m'))}_"
                                            f"{str(time.strftime('%H'))}_{str(time.strftime('%M'))}"))
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        self.path_for_run = save_folder_path

    def reset_game(self, episode_number):
        self.end_game_flag=False
        self.reset_players_positions(episode_number)
        self.win_status: WinEnum = WinEnum.NoWin

    def reset_players_positions(self, episode_number):

        is_los = True
        while is_los:
            self.blue_player._choose_random_position()
            self.red_player._choose_random_position()
            is_los = (self.red_player.x, self.red_player.y) in DICT_POS_LOS[(self.blue_player.x, self.blue_player.y)]

        if FIXED_START_POINT_RED:
            self.red_player.x = 10
            self.red_player.y = 3

        if FIXED_START_POINT_BLUE:
            self.blue_player.x = 3
            self.blue_player.y = 10


        if self.SHOW_EVERY==1 or episode_number % (self.SHOW_EVERY-1) == 0:
            self.starts_at_win_in_last_SHOW_EVERY_games = 0


    def update_win_counters(self, steps_current_game):
        if steps_current_game==MAX_STEPS_PER_EPISODE:
            self.win_array.append(WinEnum.Tie)
            self.tie_count+=1
            return

        if not self.end_game_flag:
            return

        if self.win_status == WinEnum.Red:
            self.wins_for_red += 1
            self.win_array.append(WinEnum.Red)
        elif self.win_status == WinEnum.Blue:
            self.wins_for_blue += 1
            self.win_array.append(WinEnum.Blue)
        else:
            print("Bug in update_win_counters")


    def handle_reward(self, steps_current_game):
        if not self.end_game_flag or steps_current_game==MAX_STEPS_PER_EPISODE:
            reward_step_blue = -MOVE_PENALTY
            reward_step_red = -MOVE_PENALTY
            return reward_step_blue, reward_step_red

        # Game has ended!
        if self.win_status == WinEnum.Red:
            reward_step_blue = LOST_PENALTY
            reward_step_red = WIN_REWARD

        elif self.win_status == WinEnum.Blue:
            reward_step_blue = WIN_REWARD
            reward_step_red = LOST_PENALTY

        else:
            reward_step_blue = 0
            reward_step_red = 0
            print("Bug in handle_reward- WHOS TURN?")

        return reward_step_blue, reward_step_red


    def compute_terminal(self, whos_turn=None)-> WinEnum:
        first_player = self.blue_player
        second_player = self.red_player
        win_status = WinEnum.NoWin

        is_los_first_second = (second_player.x, second_player.y) in DICT_POS_LOS[(first_player.x, first_player.y)]
        is_los_second_first = (first_player.x, first_player.y) in DICT_POS_LOS[(second_player.x, second_player.y)]
        assert is_los_first_second==is_los_second_first

        is_los = (second_player.x, second_player.y) in DICT_POS_LOS[(first_player.x, first_player.y)]
        if not is_los:  # no LOS
            win_status = WinEnum.NoWin
            self.win_status = win_status
            return win_status

        if FIRE_RANGE_FLAG:
            dist = np.linalg.norm(np.array([first_player.x, first_player.y]) - np.array([second_player.x, second_player.y]))
            if dist>FIRE_RANGE:
                win_status = WinEnum.NoWin
                self.win_status = win_status
                return win_status

        if whos_turn == Color.Blue:
            win_status = WinEnum.Blue
            self.end_game_flag = True
        elif whos_turn == Color.Red:
            win_status = WinEnum.Red
            self.end_game_flag = True
        else:
            print("Bug in compute_terminal- whos turn???")
        self.win_status = win_status
        return win_status


    def compute_terminal_with_escape(self) -> WinEnum:
        """dominating point is defined to be a point that:
         1. has LOS to the dominated_point
         2. there IS an action that executing it will end in a point that have no LOS to the second_player
         3. there is NO action for the second_player to take that will end in no LOS to the first_player

         The function will return True if the first player is dominating the second player
         False otherwise"""

        first_player = self.blue_player
        second_player = self.red_player
        win_status = WinEnum.NoWin


        is_los = (second_player.x, second_player.y) in DICT_POS_LOS[(first_player.x, first_player.y)]

        if not is_los:  # no LOS
            win_status = WinEnum.NoWin
            self.win_status = win_status
            return win_status

        if FIRE_RANGE_FLAG:
            dist = np.linalg.norm(np.array([first_player.x, first_player.y]) - np.array([second_player.x, second_player.y]))
            if dist>FIRE_RANGE:
                win_status = WinEnum.NoWin
                self.win_status = win_status
                return win_status

        can_first_escape_second, _ = can_escape(first_player, second_player)
        can_second_escape_first, _ = can_escape(second_player, first_player)

        if can_first_escape_second and not can_second_escape_first:
            win_status = WinEnum.Blue
            self.end_game_flag=True

        elif can_second_escape_first and not can_first_escape_second:
            win_status = WinEnum.Red
            self.end_game_flag = True

        else:
            win_status = WinEnum.Tie
            self.end_game_flag = True

        self.win_status = win_status
        return win_status


    def get_observation_for_blue(self)-> State:
        blue_pos = Position(self.blue_player.x, self.blue_player.y)
        red_pos = Position(self.red_player.x, self.red_player.y)
        ret_val = State(my_pos=blue_pos, enemy_pos=red_pos)

        return ret_val

    def get_observation_for_red(self)-> State:
        if not RED_PLAYER_MOVES:
            return
        blue_pos = Position(self.blue_player.x, self.blue_player.y)
        red_pos = Position(self.red_player.x, self.red_player.y)
        return State(my_pos=red_pos, enemy_pos=blue_pos)

    def take_action(self, player_color, action):
        if self.end_game_flag:
            return

        if player_color==Color.Red:
            if RED_PLAYER_MOVES:
                self.red_player.action(action)
        else: #player_color==Color.Blue
            self.blue_player.action(action)

    def can_red_win(self):
        blue_player = self.blue_player
        red_player = self.red_player
        DEBUG=False

        org_cor_blue_player_x, org_cor_blue_player_y = blue_player.get_coordinates()
        org_cor_red_player_x, org_cor_red_player_y = red_player.get_coordinates()

        ret_val = False
        winning_point_for_red = [-1, -1]
        winning_state = self.get_observation_for_blue()
        winning_action = AgentAction.Stay

        if not RED_PLAYER_MOVES:
            return ret_val, winning_state, winning_action

        for action in range(0, NUMBER_OF_ACTIONS):

            red_player.set_coordinatess(org_cor_red_player_x, org_cor_red_player_y)
            red_player.action(action)
            org_cor_blue_player_x, org_cor_blue_player_y = blue_player.get_coordinates()


            is_los = (org_cor_blue_player_x, org_cor_blue_player_y) in DICT_POS_LOS[
                (red_player.x, red_player.y)]


            if is_los:
                if FIRE_RANGE_FLAG:
                    dist = np.linalg.norm(np.array([blue_player.x, blue_player.y]) - np.array([red_player.x, red_player.y]))
                else:
                    dist = -np.inf
                if not FIRE_RANGE_FLAG or dist<=FIRE_RANGE:

                    ret_val = True

                    winning_point_for_red = (red_player.x, red_player.y)
                    blue_pos = Position(blue_player.x, blue_player.y)
                    red_pos = Position(winning_point_for_red[0], winning_point_for_red[1])
                    winning_state = State(my_pos=blue_pos, enemy_pos=red_pos)
                    # Red Takes winning move!!!
                    return ret_val, winning_state, AgentAction(action)

        red_player.set_coordinatess(org_cor_red_player_x, org_cor_red_player_y)
        if DEBUG:
            red_player.set_coordinatess(org_cor_red_player_x, org_cor_red_player_y)
            import matplotlib.pyplot as plt
            blue_obs_satrt = self.get_observation_for_blue()
            plt.matshow(blue_obs_satrt.img)
            plt.show()

            blue_pos = Position(blue_player.x, blue_player.y)
            red_pos = Position(winning_point_for_red[0], winning_point_for_red[1])
            winning_state = State(my_pos=blue_pos, enemy_pos=red_pos)
            plt.matshow(winning_state.img)
            plt.show()

        return ret_val, winning_state, winning_action


    def can_blue_win(self):
        blue_player = self.blue_player
        red_player = self.red_player
        DEBUG=False

        org_cor_blue_player_x, org_cor_blue_player_y = blue_player.get_coordinates()
        org_cor_red_player_x, org_cor_red_player_y = red_player.get_coordinates()

        ret_val = False
        winning_point_for_blue = [-1, -1]
        winning_state = self.get_observation_for_red()
        winning_action = AgentAction.Stay


        for action in range(0, NUMBER_OF_ACTIONS):

            blue_player.set_coordinatess(org_cor_blue_player_x, org_cor_blue_player_y)
            blue_player.action(action)
            org_cor_red_player_x, org_cor_red_player_y = red_player.get_coordinates()

            is_los = (org_cor_red_player_x, org_cor_red_player_y) in DICT_POS_LOS[
                (blue_player.x, blue_player.y)]


            if is_los:
                if FIRE_RANGE_FLAG:
                    dist = np.linalg.norm(np.array([blue_player.x, blue_player.y]) - np.array([red_player.x, red_player.y]))
                else:
                    dist = -np.inf
                if not FIRE_RANGE_FLAG or dist<=FIRE_RANGE:

                    ret_val = True

                    winning_point_for_blue = (blue_player.x, blue_player.y)
                    red_pos = Position(red_player.x, red_player.y)
                    blue_pos = Position(winning_point_for_blue[0], winning_point_for_blue[1])
                    winning_state = State(my_pos=red_pos, enemy_pos=blue_pos)
                    # Red Takes winning move!!!
                    return ret_val, winning_state, AgentAction(action)

        blue_player.set_coordinatess(org_cor_blue_player_x, org_cor_blue_player_y)
        if DEBUG:
            red_player.set_coordinatess(org_cor_red_player_x, org_cor_red_player_y)
            import matplotlib.pyplot as plt
            red_obs_satrt = self.get_observation_for_red()
            plt.matshow(red_obs_satrt.img)
            plt.show()

            red_pos = Position(red_player.x, red_player.y)
            blue_pos = Position(winning_point_for_blue[0], winning_point_for_blue[1])
            winning_state = State(my_pos=blue_pos, enemy_pos=red_pos)
            plt.matshow(winning_state.img)
            plt.show()

        return ret_val, winning_state, winning_action



    def end_run(self):
        STATS_RESULTS_RELATIVE_PATH_THIS_RUN = os.path.join(self.path_for_run, STATS_RESULTS_RELATIVE_PATH)
        self.save_folder_path = path.join(STATS_RESULTS_RELATIVE_PATH_THIS_RUN,
                                     format(f"{str(time.strftime('%d'))}_{str(time.strftime('%m'))}_"
                                            f"{str(time.strftime('%H'))}_{str(time.strftime('%M'))}_{Agent_type_str[self.blue_player._decision_maker.type()]}_{Agent_type_str[self.red_player._decision_maker.type()]}_{str(STR_FOLDER_NAME)}"))

        # save info on run
        self.save_stats(self.save_folder_path)

        # print and save figures
        print_stats(self.episodes_rewards_blue, self.save_folder_path, self.SHOW_EVERY, player=Color.Blue)
        print_stats(self.episodes_rewards_red, self.save_folder_path, self.SHOW_EVERY, player=Color.Red)
        print_stats(self.steps_per_episode, self.save_folder_path,self.SHOW_EVERY, save_figure=True, steps=True, player=Color.Blue)

        save_reward_stats(self.save_folder_path, self.SHOW_EVERY, self.episodes_rewards_blue, self.episodes_rewards_red, self.steps_per_episode, self.blue_epsilon_values)

        save_win_statistics(self.win_array, self.save_folder_path, self.SHOW_EVERY)

    def data_for_statistics(self, episode_reward_blue, episode_reward_red, steps_current_game, blue_epsilon):
        self.episodes_rewards_blue.append(episode_reward_blue)
        self.episodes_rewards_red.append(episode_reward_red)
        self.steps_per_episode.append(steps_current_game)
        self.blue_epsilon_values.append(blue_epsilon)


    def save_stats(self, save_folder_path):

        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        chcek_unvisited_states = False
        counter_ones = 0
        num_of_states = 15 * 15 * 15 * 15
        if self.red_player._decision_maker.type() == AgentType.Q_table:
            Q_matrix = self.red_player._decision_maker._Q_matrix
            chcek_unvisited_states = True
        elif self.blue_player._decision_maker.type() == AgentType.Q_table:
            Q_matrix = self.blue_player._decision_maker._Q_matrix
            chcek_unvisited_states = True
        if chcek_unvisited_states:
            num_of_states = 15 * 15 * 15 * 15
            block_states = np.sum(DSM)
            for x1 in range(SIZE_Y):
                for y1 in range(SIZE_Y):
                    for x2 in range(SIZE_Y):
                        for y2 in range(SIZE_Y):
                            if list(Q_matrix[(x1, y1), (x2, y2)]) == list(np.ones(NUMBER_OF_ACTIONS)):
                                counter_ones += 1

        print("for", self.NUMBER_OF_EPISODES, "episodes: ")

        info = {f"NUM_OF_EPISODES": [NUM_OF_EPISODES],
                f"USE_LOS_IN_STATE": [USE_LOS_IN_STATE],
                f"MOVE_PENALTY": [MOVE_PENALTY],
                f"WIN_REWARD": [WIN_REWARD],
                f"LOST_PENALTY": [LOST_PENALTY],
                f"epsilon": [START_EPSILON],
                f"EPSILONE_DECAY": [EPSILONE_DECAY],
                f"LEARNING_RATE": [LEARNING_RATE],
                f"DISCOUNT": [DISCOUNT],
                f"IMG_STATE_MODE": [IMG_STATE_MODE],
                f"ACTION_SPACE_9": [ACTION_SPACE_9],
                f"DANGER_ZONE_IN_STATE": [DANGER_ZONE_IN_STATE],
                f"DOMINATING_POINTS_IN_STATE": [DOMINATING_POINTS_IN_STATE],
                f"FIRE_RANGE_FLAG": [FIRE_RANGE_FLAG],
                f"FIRE_RANGE": [FIRE_RANGE],
                f"DSM_NAME": [DSM_name],
                f"RED_PLAYER_MOVES": [RED_PLAYER_MOVES],
                f"FIXED_START_POINT_RED": [FIXED_START_POINT_RED],
                f"FIXED_START_POINT_BLUE": [FIXED_START_POINT_BLUE],
                f"%Games started at Tie" : [self.starts_at_win / self.NUMBER_OF_EPISODES*100],
                f"%WINS_BLUE": [self.wins_for_blue/self.NUMBER_OF_EPISODES*100],
                f"%WINS_RED": [self.wins_for_red/self.NUMBER_OF_EPISODES*100],
                f"%TIES": [self.tie_count/self.NUMBER_OF_EPISODES*100],
                f"%Blue_agent_type" : [Agent_type_str[self.blue_player._decision_maker.type()]],
                f"%Blue_agent_model_loded": [self.blue_player._decision_maker.path_model_to_load],
                f"%Red_agent_type" : [Agent_type_str[self.red_player._decision_maker.type()]],
                f"%Red_agent_model_loded": [self.red_player._decision_maker.path_model_to_load]}


        df = pd.DataFrame(info)
        df.to_csv(os.path.join(save_folder_path, 'Statistics.csv'), index=False)

        # save models
        self.red_player._decision_maker.save_model(self.episodes_rewards_blue, save_folder_path, Color.Red)
        self.blue_player._decision_maker.save_model(self.episodes_rewards_blue, save_folder_path, Color.Blue)


class Episode():
    def __init__(self, episode_number, show_always=False):
        self.episode_number = episode_number
        self.episode_reward_blue = 0
        self.episode_reward_red = 0
        self.is_terminal = False

        if episode_number % SHOW_EVERY == 0 or episode_number == 1 or show_always:
            self.show = True
        else:
            self.show = False

        self.Blue_starts = True#np.random.random() >= 0.5 # for even statistics

    def whos_turn(self, steps_current_game)-> Color:
        if self.Blue_starts:
            if steps_current_game % 2 == 1:
                return Color.Blue
            else:
                return Color.Red

        else:
            if steps_current_game % 2 == 1:
                return Color.Red
            else:
                return Color.Blue

    def print_episode(self, env, last_step_number, save_file=False):
        if self.show and USE_DISPLAY:
            print_episode_graphics(env, self, last_step_number, save_file)

    def get_image(self, env, image_for_red = False):
        image = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8) # starts an rbg of small world
        if image_for_red: #switch the blue and red colors
            image[env.red_player.x][env.red_player.y] = dict_of_colors_for_graphics[BLUE_N]
            image[env.blue_player.x][env.blue_player.y] = dict_of_colors_for_graphics[RED_N]
        else:
            image[env.blue_player.x][env.blue_player.y] = dict_of_colors_for_graphics[RED_N]
            image[env.red_player.x][env.red_player.y] = dict_of_colors_for_graphics[BLUE_N]
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x][y] == 1.:
                    image[x][y] = dict_of_colors_for_graphics[GREY_N]
        img = Image.fromarray(image, 'RGB')
        # img = img.resize((600, 600))
        # Image._show(img)
        return img


    def print_info_of_episode(self, env, steps_current_game, blue_epsilon, EVALUATE=False):
        if self.show:
            if len(env.episodes_rewards_blue)<env.SHOW_EVERY:
                number_of_episodes = len(env.episodes_rewards_blue[-env.SHOW_EVERY:]) - 1
            else:
                number_of_episodes = env.SHOW_EVERY

            print(f"\non #{self.episode_number}:")

            print(f"reward for blue player is: , {self.episode_reward_blue}")
            print(f"epsilon (blue player) is {blue_epsilon}")
            print(
                f"mean number of steps of last {number_of_episodes} episodes: , {np.mean(env.steps_per_episode[-env.SHOW_EVERY:])}")

            print(f"mean rewards of last {number_of_episodes} episodes for blue player: {np.mean(env.episodes_rewards_blue[-env.SHOW_EVERY:])}")

            win_array = np.array(env.win_array[-env.SHOW_EVERY:])
            blue_win_per_for_last_games = np.sum(win_array==WinEnum.Blue) / number_of_episodes * 100
            red_win_per_for_last_games = np.sum(win_array == WinEnum.Red) / number_of_episodes * 100
            ties_LOS = np.sum(win_array==WinEnum.Tie) / number_of_episodes * 100
            ties_num_of_steps = np.sum(win_array == WinEnum.NoWin) / number_of_episodes * 100


            print(f"in the last {number_of_episodes} episodes, BLUE won: {blue_win_per_for_last_games}%, RED won {red_win_per_for_last_games}%, ended in TIE do to LOS: {ties_LOS}%, ended in TIE do to steps: {ties_num_of_steps}% of games")

            print(f"mean rewards of all episodes for blue player: {np.mean(env.episodes_rewards_blue)}\n")

            self.print_episode(env, steps_current_game)

        if EVALUATE:
            print("\nEvaluation summury: num_episodes: ", self.episode_number, ", number of steps: ", steps_current_game)
            print("reward for blue: ",  self.episode_reward_blue)
            print("reward for red: ", self.episode_reward_red)


        if self.episode_number % SAVE_STATS_EVERY == 0:
            env.end_run()

