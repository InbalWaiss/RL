
from matplotlib import style
from tqdm import tqdm

style.use("ggplot")
from Arena.CState import State
from Arena.Entity import Entity
from Arena.Environment import Environment, Episode
from Common.constants import *
from Qtable import Qtable_DecisionMaker
from DQN import DQNAgent_keras
from Greedy import Greedy_player


def print_start_of_game_info(blue_decision_maker, red_decision_maker):
    print("Starting tournament!")
    print("Blue player type: ", Agent_type_str[blue_decision_maker.type()])
    if blue_decision_maker.path_model_to_load==None:
        print("Blue player starting with no model")
    else:
        print("Blue player starting tournament with trained model: " , blue_decision_maker.path_model_to_load)

    print("Red player type: ", Agent_type_str[red_decision_maker.type()])
    if red_decision_maker.path_model_to_load==None:
        print("Red player starting with no model")
    else:
        print("Red player starting tournament with trained model: " , red_decision_maker.path_model_to_load)


    print("Number of rounds: ", NUM_OF_EPISODES)
    print("~~~ GO! ~~~\n\n")


def evaluate(episode_number):
    if episode % EVALUATE_PLAYERS_EVERY == 0:
        EVALUATE = True
    else:
        EVALUATE = False
    return EVALUATE


def print_observation(observation_for_blue_s0, observation_for_blue_s1):
    import matplotlib.pyplot as plt

    plt.matshow(observation_for_blue_s0.img)
    plt.show()
    plt.matshow(observation_for_blue_s1.img)
    plt.show()



# MAIN:
if __name__ == '__main__':

    env = Environment(IS_TRAINING)

    ### Red Decision Maker
    #red_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker()
    red_decision_maker = Greedy_player.Greedy_player()
    #red_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker(UPDATE_CONTEXT=False , path_model_to_load="qtable_red-1000000_penalty_move_-1.pickle")
    #red_decision_maker = DQNAgent_keras.DQNAgent_keras(UPDATE_CONTEXT=False,
    #                                                    path_model_to_load='flatten_FC1-elu_FC2-elu_FC3-elu_FC4-elu__red_25001_  -6.00max_ -72.99avg_-100.00min__1615541339.model')
    #red_decision_maker = DQNAgent_keras.DQNAgent_keras()

    ### Blue Decision Maker
    #--Greedy:
    #blue_decision_maker = Greedy_player.Greedy_player()
    # --Qtable:
    #blue_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker()
    #blue_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker('qtable_blue-600000_penalty_move_-1.pickle')
    # --DQN Basic:
    #blue_decision_maker = DQNAgent.DQNAgent()
    # blue_decision_maker = DQNAgent.DQNAgent(UPDATE_CONTEXT=False, path_model_to_load='basic_DQN_17500_blue.model')
    # --DQN Keras
    blue_decision_maker = DQNAgent_keras.DQNAgent_keras()
    #blue_decision_maker = DQNAgent_keras.DQNAgent_keras(UPDATE_CONTEXT=True, path_model_to_load='flatten_FC1-elu_FC2-elu_FC3-elu_FC4-elu__blue_800001_ 116.00max_  -6.80avg_-361.00min__1616132768.model')
    #flatten_FC1-elu_FC2-elu_FC3-elu_FC4-elu__blue_30001_ 120.00max_  97.59avg_-100.00min__1615828123
    # blue_decision_maker = DQNAgent_keras.DQNAgent_keras(UPDATE_CONTEXT=True,
    #                                                     path_model_to_load='flatten_FC1-elu_FC2-elu_FC3-elu_FC4-elu__blue_157501_ 120.00max_   4.80avg_   0.00min__1615952583.model')

    # --DQN Attention
    # blue_decision_maker = DQNAgent_spatioalAttention.DQNAgent_spatioalAttention()
    # blue_decision_maker = DQNAgent_spatioalAttention.DQNAgent_spatioalAttention(UPDATE_CONTEXT=True, path_model_to_load='statistics/18_02_06_54_DQNAgent_spatioalAttention_Q_table_1000000/qnet1000000.cptk')
    # blue_decision_maker = DQNAgent_temporalAttention.DQNAgent_temporalAttention()


    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)

    print_start_of_game_info(blue_decision_maker, red_decision_maker)

    NUM_OF_EPISODES = env.NUMBER_OF_EPISODES
    for episode in tqdm(range(1, NUM_OF_EPISODES + 1), ascii=True, unit='episodes'):

        current_episode = Episode(episode, show_always=False if IS_TRAINING else True)

        # set new start position for the players
        env.reset_players_positions(episode)

        # get observation
        observation_for_blue_s0: State = env.get_observation_for_blue()


        action_blue = -1#AgentAction(np.random.randint(0, NUMBER_OF_ACTIONS))


        # initialize the decision_makers for the players
        blue_decision_maker.set_initial_state(observation_for_blue_s0, episode)

        EVALUATE = evaluate(episode)

        blue_won_the_game = False
        red_won_the_game = False
        steps_current_game = 0
        for steps_current_game in range(1, MAX_STEPS_PER_EPISODE + 1):

            ##### Blue's turn! #####
            observation_for_blue_s0: State = env.get_observation_for_blue()
            current_episode.print_episode(env, steps_current_game)

            action_blue: AgentAction = blue_decision_maker.get_action(observation_for_blue_s0, EVALUATE)
            env.blue_player.action(action_blue)  # take the action!
            current_episode.print_episode(env, steps_current_game)

            current_episode.is_terminal = (env.compute_terminal(whos_turn=Color.Blue) is not WinEnum.NoWin)
            if current_episode.is_terminal:
                blue_won_the_game = True

            if not current_episode.is_terminal:
                ##### Red's turn! #####
                observation_for_red_s0: State = env.get_observation_for_red()
                action_red: AgentAction = red_decision_maker.get_action(observation_for_red_s0, EVALUATE)
                env.red_player.action(action_red)  # take the action!
                current_episode.is_terminal = (env.compute_terminal(whos_turn=Color.Red) is not WinEnum.NoWin)
                current_episode.print_episode(env, steps_current_game)
                if current_episode.is_terminal:
                    red_won_the_game = True


            observation_for_blue_s1: State = env.get_observation_for_blue()

            reward_step_blue, reward_step_red = env.handle_reward(steps_current_game,
                                                                  current_episode.is_terminal)
            if current_episode.is_terminal:
                if blue_won_the_game:
                    reward_step_blue, reward_step_red = env.handle_reward(steps_current_game,
                                                                          current_episode.is_terminal,
                                                                          whos_turn=Color.Blue)
                    env.update_win_counters(steps_current_game, whos_turn=Color.Blue)
                elif red_won_the_game:
                    reward_step_blue, reward_step_red = env.handle_reward(steps_current_game,
                                                                          current_episode.is_terminal,
                                                                          whos_turn=Color.Red)
                    env.update_win_counters(steps_current_game, whos_turn=Color.Red)



            blue_decision_maker.update_context(observation_for_blue_s0, action_blue, reward_step_blue, observation_for_blue_s1,
                                               current_episode.is_terminal, EVALUATE)

            current_episode.episode_reward_red += reward_step_red
            current_episode.episode_reward_blue += reward_step_blue


            if steps_current_game == MAX_STEPS_PER_EPISODE:
                # if we exited the loop because we reached MAX_STEPS_PER_EPISODE
                current_episode.is_terminal = True
                env.update_win_counters(steps_current_game)

            if current_episode.is_terminal:
                break




        # for statistics
        env.data_for_statistics(current_episode.episode_reward_blue, current_episode.episode_reward_red, steps_current_game, blue_decision_maker.get_epsolon())

        # print info of episode:
        current_episode.print_info_of_episode(env, steps_current_game, blue_decision_maker.get_epsolon())

    env.end_run()


