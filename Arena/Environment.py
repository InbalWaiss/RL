
import time

from RafaelPlayer.Qtable_DecisionMaker import *
from RafaelPlayer.QPlayer_constants import START_EPSILON, EPSILONE_DECAY, LEARNING_RATE, DISCOUNT
from Arena.Position import Position
from Arena.graphics import print_stats, print_episode_graphics
from Arena.helper_funcs import *
import numpy as np
from PIL import Image
import pandas as pd


class Environment(object):
    def __init__(self, TRAIN=True):
        self.episodes_rewards = []
        self.episodes_rewards.append(0)
        self.steps_per_episode = []
        self.steps_per_episode.append(0)
        self.number_of_steps = 0
        self.wins_for_blue = 0
        self.wins_for_red = 0
        self.tie_count = 0
        self.starts_at_win = 0
        self.starts_at_win_in_last_SHOW_EVERY_games = 0
        self.win_status: WinEnum = WinEnum.NoWin

        self.blue_player = None
        self.red_player = None

        if TRAIN:
            self.SHOW_EVERY = SHOW_EVERY
            self.NUMBER_OF_EPISODES = NUM_OF_EPISODES

        else:
            self.SHOW_EVERY = EVALUATE_SHOW_EVERY
            self.NUMBER_OF_EPISODES = EVALUATE_NUM_OF_EPISODES

        self.create_path_for_statistics()

    def create_path_for_statistics(self):
        save_folder_path = path.join(STATS_RESULTS_RELATIVE_PATH,
                                     format(f"{str(time.strftime('%d'))}_{str(time.strftime('%m'))}_"
                                            f"{str(time.strftime('%H'))}_{str(time.strftime('%M'))}"))
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        self.path_for_run = save_folder_path

    def reset_players_positions(self, episode_number):
        self.blue_player._choose_random_position()
        self.red_player._choose_random_position()

        if self.SHOW_EVERY==1 or episode_number % (self.SHOW_EVERY-1) == 0:
            self.starts_at_win_in_last_SHOW_EVERY_games = 0

    def update_win_counters(self, steps_current_game):

        if self.win_status == WinEnum.Blue:
            self.wins_for_blue += 1
        elif self.win_status == WinEnum.Red:
            self.wins_for_red += 1
        elif self.win_status == WinEnum.Tie:
            self.tie_count += 1

    def handle_reward(self, steps_current_game):

        reward_value = WIN_REWARD - steps_current_game * MOVE_PENALTY
        if self.win_status == WinEnum.Blue:
            reward = reward_value
        elif self.win_status == WinEnum.Red:
            reward = -1 * reward_value
        else:
            reward = 0

        reward_blue = reward
        reward_red = -reward
        return reward_blue, reward_red

    def compute_terminal(self) -> WinEnum:
        """dominating point is defined to be a point that:
         1. has LOS to the dominated_point
         2. there IS an action that executing it will end in a point that have no LOS to the second_player
         3. there is NO action for the second_player to take that will end in no LOS to the first_player

         The function will return True if the first player is dominating the second player
         False otherwise"""

        first_player = self.blue_player
        second_player = self.red_player
        win_status = WinEnum.NoWin
        is_los, _ = check_if_LOS(first_player.x, first_player.y, second_player.x, second_player.y)
        if not is_los:  # no LOS
            win_status = WinEnum.NoWin
            self.win_status = win_status
            return win_status

        can_first_escape_second, _ = can_escape(first_player, second_player)
        can_second_escape_first, _ = can_escape(second_player, first_player)

        if can_first_escape_second and not can_second_escape_first:
            win_status = WinEnum.Blue

        elif can_second_escape_first and not can_first_escape_second:
            win_status = WinEnum.Red

        else:
            win_status = WinEnum.Tie

        self.win_status = win_status
        return win_status


    def get_observation_for_blue(self)-> State:

        blue_pos = Position(self.blue_player.x, self.blue_player.y)
        red_pos = Position(self.red_player.x, self.red_player.y)
        ret_val = State(my_pos=blue_pos, enemy_pos=red_pos)

        return ret_val

    def get_observation_for_red(self)-> State:

        blue_pos = Position(self.blue_player.x, self.blue_player.y)
        red_pos = Position(self.red_player.x, self.red_player.y)
        return State(my_pos=red_pos, enemy_pos=blue_pos)

    def end_run(self):
        STATS_RESULTS_RELATIVE_PATH_THIS_RUN = os.path.join(self.path_for_run, STATS_RESULTS_RELATIVE_PATH)
        save_folder_path = path.join(STATS_RESULTS_RELATIVE_PATH_THIS_RUN,
                                     format(f"{str(time.strftime('%d'))}_{str(time.strftime('%m'))}_"
                                            f"{str(time.strftime('%H'))}_{str(time.strftime('%M'))}_{Agent_type_str[self.blue_player._decision_maker.type()]}_{Agent_type_str[self.red_player._decision_maker.type()]}"))

        # save info on run
        self.save_stats(save_folder_path)

        # print and save figures
        print_stats(self.episodes_rewards, save_folder_path, self.SHOW_EVERY)
        print_stats(self.steps_per_episode, save_folder_path,self.SHOW_EVERY, True, True)



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
        if chcek_unvisited_states:
            print("% of unseen states: ", counter_ones / (num_of_states-block_states) * 100)
        print("% of games started at win: ", self.starts_at_win / self.NUMBER_OF_EPISODES * 100)

        info = {f"NUM_OF_EPISODES": [NUM_OF_EPISODES],
                f"USE_LOS_IN_STATE": [USE_LOS_IN_STATE],
                f"MOVE_PENALTY": [MOVE_PENALTY],
                f"WIN_REWARD": [WIN_REWARD],
                f"LOST_PENALTY": [LOST_PENALTY],
                f"epsilon": [START_EPSILON],
                f"EPSILONE_DECAY": [EPSILONE_DECAY],
                f"LEARNING_RATE": [LEARNING_RATE],
                f"DISCOUNT": [DISCOUNT],
                f"% Unseen states": [counter_ones / num_of_states * 100],
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
        self.red_player._decision_maker.save_model(self.episodes_rewards, save_folder_path, Color.Red)
        self.blue_player._decision_maker.save_model(self.episodes_rewards, save_folder_path, Color.Blue)


class Episode():
    def __init__(self, episode_number, show_always=False):
        self.episode_number = episode_number
        self.episode_reward_blue = 0
        self.episode_reward_blue_array = []
        self.is_terminal = False

        if episode_number % SHOW_EVERY == 0 or episode_number == 1 or show_always:
            self.show = True
        else:
            self.show = False

    def print_episode(self, env, last_step_number, save_file=False):
        if self.show:
            print_episode_graphics(env, self, last_step_number, save_file)

    def get_image(self, env, image_for_red = False):
        image = np.zeros((SIZE_X, SIZE_Y, 3), dtype=np.uint8) # starts an rbg of small world
        if image_for_red: #switch the blue and red colors
            image[env.red_player.x][env.red_player.y] = dict_of_colors[BLUE_N]
            image[env.blue_player.x][env.blue_player.y] = dict_of_colors[RED_N]
        else:
            image[env.blue_player.x][env.blue_player.y] = dict_of_colors[RED_N]
            image[env.red_player.x][env.red_player.y] = dict_of_colors[BLUE_N]
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x][y] == 1.:
                    image[x][y] = dict_of_colors[GREY_N]
        img = Image.fromarray(image, 'RGB')
        # img = img.resize((600, 600))
        # Image._show(img)
        return img


    def print_info_of_episode(self, env, steps_current_game, blue_epsilon):
        if self.show:
            if len(env.episodes_rewards)<env.SHOW_EVERY:
                number_of_episodes = len(env.episodes_rewards[-env.SHOW_EVERY:]) - 1
            else:
                number_of_episodes = env.SHOW_EVERY

            print(f"\non #{self.episode_number}:")

            print(f"reward for blue player is: , {self.episode_reward_blue}")
            print(f"epsilon (blue player) is {blue_epsilon}")
            print(
                f"mean number of steps of last {number_of_episodes} episodes: , {np.mean(env.steps_per_episode[-env.SHOW_EVERY:])}")

            print(f"mean rewards of last {number_of_episodes} episodes for blue player: {np.mean(env.episodes_rewards[-env.SHOW_EVERY:])}")

            blue_win_per_for_last_games = np.sum(np.array(env.episodes_rewards[-env.SHOW_EVERY:])>0)/number_of_episodes*100
            tie_per_for_last_games = (np.sum(np.array(env.episodes_rewards[-env.SHOW_EVERY:])==0)-1)/number_of_episodes*100
            red_win_per_for_last_games = np.sum(np.array(env.episodes_rewards[-env.SHOW_EVERY:])<0)/number_of_episodes*100
            print(f"in the last {number_of_episodes} episodes, BLUE won: {blue_win_per_for_last_games}%, RED won {red_win_per_for_last_games}%, ended in TIE: {tie_per_for_last_games}%, started at TIE: {env.starts_at_win_in_last_SHOW_EVERY_games}% of games")

            print(f"mean rewards of all episodes for blue player: {np.mean(env.episodes_rewards)}\n")

            self.print_episode(env, steps_current_game)


        if self.episode_number % SAVE_STATS_EVERY == 0:
            env.end_run()

