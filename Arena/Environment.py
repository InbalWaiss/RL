from RafaelPlayer.RafaelDecisionMaker import *
from RafaelPlayer.QPlayer_constants import START_EPSILON, EPSILONE_DECAY, LEARNING_RATE, DISCOUNT
from Arena.Position import Position
from Arena.graphics import print_stats, print_episode_graphics
from Arena.helper_funcs import *
import numpy as np
from PIL import Image


class Environment(object):
    def __init__(self):
        self.episodes_rewards = []
        self.episodes_rewards.append(0)
        self.steps_per_episode = []
        self.steps_per_episode.append(0)
        self.number_of_steps = 0
        self.wins_for_blue = 0
        self.wins_for_red = 0
        self.tie_count = 0
        self.starts_at_win = 0

        self.blue_player = None
        self.red_player = None
        self.SHOW_EVERY = SHOW_EVERY
        self.NUMBER_OF_EPISODES = NUM_OF_EPISODES

    def update_win_counters(self):
        reward_blue, reward_red = self.handle_reward()
        if reward_blue == WIN_REWARD:
            self.wins_for_blue += 1
        elif reward_red == WIN_REWARD:
            self.wins_for_red += 1
        else:
            self.tie_count += 1


    def handle_reward(self):
        # handle the rewarding
        blue_dominate_red, _ = is_dominating(self.blue_player, self.red_player)
        red_dominate_blue, _ = is_dominating(self.red_player, self.blue_player)
        reward = 0
        if blue_dominate_red and red_dominate_blue:
            reward = TIE
        elif blue_dominate_red:
            reward = WIN_REWARD
        elif red_dominate_blue:
            reward = -WIN_REWARD

        reward_blue = reward
        reward_red = -reward
        return reward_blue, reward_red

    def check_terminal(self):
        flag_red_on_blue, _ = is_dominating(self.red_player, self.blue_player)
        flag_blue_on_red, _ = is_dominating(self.blue_player, self.red_player)
        if self.blue_player.x == self.red_player.x and self.blue_player.y == self.red_player.y:
            return True
        elif flag_red_on_blue:
            return True
        elif flag_blue_on_red:
            return True
        return False

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
        save_folder_path = path.join(STATS_RESULTS_RELATIVE_PATH,
                                     format(f"{str(time.strftime('%d'))}_{str(time.strftime('%m'))}_"
                                            f"{str(time.strftime('%H'))}_{str(time.strftime('%M'))}"))

        # save info on run
        self.save_stats(save_folder_path)

        # print and save figures
        print_stats(self.episodes_rewards, save_folder_path, self.SHOW_EVERY)
        print_stats(self.steps_per_episode, save_folder_path,self.SHOW_EVERY, True, True)



    def save_stats(self, save_folder_path):
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        chcek_unvisited_states = False
        if self.red_player._decision_maker.type() == AgentType.Q_table:
            Q_matrix = self.red_player._decision_maker._Q_matrix
            chcek_unvisited_states = True
        elif self.blue_player._decision_maker.type() == AgentType.Q_table:
            Q_matrix = self.blue_player._decision_maker._Q_matrix
            chcek_unvisited_states = True
        if chcek_unvisited_states:
            num_of_states = 15 * 15 * 15 * 15
            block_states = np.sum(DSM)
            counter_ones = 0
            for x1 in range(SIZE_Y):
                for y1 in range(SIZE_Y):
                    for x2 in range(SIZE_Y):
                        for y2 in range(SIZE_Y):
                            if list(Q_matrix[(x1, y1), (x2, y2)]) == list(np.ones(NUMBER_OF_ACTIONS)):
                                counter_ones += 1

        print("for", NUM_OF_EPISODES, "episodes: ")
        if chcek_unvisited_states:
            print("% of unseen states: ", counter_ones / (num_of_states-block_states) * 100)
        print("% of games started at win: ", self.starts_at_win / self.NUMBER_OF_EPISODES * 100)

        info = {f"NUM_OF_EPISODES": [NUM_OF_EPISODES],
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
        self.red_player._decision_maker.save_model(self.NUMBER_OF_EPISODES, save_folder_path, Color.Red)
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

    def print_episode(self, env, last_step_number):
        if self.show:
            print_episode_graphics(env, self, last_step_number)

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
            print(f"on #{self.episode_number}:")
            print(f"reward for blue player is: , {self.episode_reward_blue}")
            print(f"epsilon (blue player) is {blue_epsilon}")
            print(
                f"mean rewards of last {SHOW_EVERY} episodes for blue player: {np.mean(env.episodes_rewards[-SHOW_EVERY:])}")
            print(
                f"mean rewards of all episodes for blue player: {np.mean(env.episodes_rewards)}")
            print(f"mean number of steps: , {np.mean(env.steps_per_episode[-SHOW_EVERY:])}\n")
            self.print_episode(env, steps_current_game)

