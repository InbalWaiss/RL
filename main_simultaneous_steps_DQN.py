
from matplotlib import style
from tqdm import tqdm

style.use("ggplot")
from Arena.CState import State
from Arena.Entity import Entity
from Arena.Environment import Environment, Episode
from Arena.constants import *
from RafaelPlayer import Qtable_DecisionMaker
from DQN import DQNAgent_keras, DQNAgent_temporalAttention, DQNAgent_spatioalAttention
from DQN import DQNAgent




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


# MAIN:
if __name__ == '__main__':

    env = Environment(IS_TRAINING)

    ### Red Decision Maker
    # red_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker()
    red_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker(UPDATE_CONTEXT=False , path_model_to_load="qtable_red-1000000_old_terminal_state.pickle")


    ### Blue Decision Maker
    # --Qtable:
    # blue_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker()
    # blue_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker('qtable_blue-1000000_old_terminal_state.pickle')
    # --DQN Basic:
    # blue_decision_maker = DQNAgent.DQNAgent()
    # blue_decision_maker = DQNAgent.DQNAgent(UPDATE_CONTEXT=False, path_model_to_load='basic_DQN_17500_blue.model')
    # --DQN Keras
    blue_decision_maker = DQNAgent_keras.DQNAgent_keras()
    #blue_decision_maker = DQNAgent_keras.DQNAgent_keras(path_model_to_load='flatten_FC1-elu_FC2-elu_FC3-elu__blue_22501_   1.00max_ -31.34avg_-250.00min__1615210919.model')
    # --DQN Attention
    # blue_decision_maker = DQNAgent_spatioalAttention.DQNAgent_spatioalAttention()
    # blue_decision_maker = DQNAgent_spatioalAttention.DQNAgent_spatioalAttention(UPDATE_CONTEXT=True, path_model_to_load='statistics/18_02_06_54_DQNAgent_spatioalAttention_Q_table_1000000/qnet1000000.cptk')
    # blue_decision_maker = DQNAgent_temporalAttention.DQNAgent_temporalAttention()

    print("np.clip(reward, -1, 1)")
    print("WIN_REWARD = 100")

    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)

    print_start_of_game_info(blue_decision_maker, red_decision_maker)

    NUM_OF_EPISODES = env.NUMBER_OF_EPISODES
    for episode in tqdm(range(1, NUM_OF_EPISODES + 1), ascii=True, unit='episodes'):

        current_episode = Episode(episode, show_always=False if IS_TRAINING else True)

        # set new start position for the players
        env.reset_players_positions(episode)

        if FIXED_START_POINT_RED:
            env.red_player.x = 10
            env.red_player.y = 3

        if FIXED_START_POINT_BLUE:
            env.blue_player.x = 3
            env.blue_player.y = 10

        # get observation
        initial_state_blue: State = env.get_observation_for_blue()
        initial_state_red: State = env.get_observation_for_red()

        # initialize the decision_makers for the players
        blue_decision_maker.set_initial_state(initial_state_blue, episode)
        red_decision_maker.set_initial_state(initial_state_red, episode)

        for steps_current_game in range(1, MAX_STEPS_PER_EPISODE + 1):

            env.number_of_steps += 1

            # get observation
            observation_for_blue: State = env.get_observation_for_blue()
            if RED_PLAYER_MOVES:
                observation_for_red: State = env.get_observation_for_red()

            # Check if the start state is terminal
            current_episode.is_terminal = (env.compute_terminal() is not WinEnum.NoWin)
            if current_episode.is_terminal:
                env.update_win_counters(steps_current_game)
                env.starts_at_win += 1
                env.starts_at_win_in_last_SHOW_EVERY_games +=1
                reward_step_blue, reward_step_red = env.handle_reward(steps_current_game)
                current_episode.episode_reward_blue += reward_step_blue
                current_episode.episode_reward_red += reward_step_red
                break

            ##### Blue's turn! #####
            action_blue: AgentAction = blue_decision_maker.get_action(observation_for_blue)
            env.blue_player.action(action_blue)  # take the action!

            if RED_PLAYER_MOVES:
                ##### Red's turn! #####
                action_red: AgentAction = red_decision_maker.get_action(observation_for_red)
                env.red_player.action(action_red) # take the action!

            # Get new observations
            new_observation_for_blue: State = env.get_observation_for_blue()
            if RED_PLAYER_MOVES:
                new_observation_for_red: State = env.get_observation_for_red()

            # Check if terminal
            current_episode.is_terminal = (env.compute_terminal() is not WinEnum.NoWin)

            # Handle rewards
            reward_step_blue, reward_step_red = env.handle_reward(steps_current_game)
            current_episode.episode_reward_blue += reward_step_blue
            current_episode.episode_reward_red += reward_step_red

            # Update models
            blue_decision_maker.update_context(new_observation_for_blue,
                                                  reward_step_blue,
                                                  current_episode.is_terminal)

            if RED_PLAYER_MOVES:
                red_decision_maker.update_context(new_observation_for_red,
                                                      reward_step_red,
                                                      current_episode.is_terminal)

            current_episode.print_episode(env, steps_current_game)
            if current_episode.is_terminal:
                env.update_win_counters(steps_current_game)
                break

        if steps_current_game == MAX_STEPS_PER_EPISODE:
            # if we exited the loop because we reached MAX_STEPS_PER_EPISODE
            env.update_win_counters(steps_current_game)
            current_episode.is_terminal = True

        # for statistics
        env.data_for_statistics(current_episode.episode_reward_blue, current_episode.episode_reward_red, steps_current_game, blue_decision_maker.get_epsolon())

        # print info of episode:
        current_episode.print_info_of_episode(env, steps_current_game, blue_decision_maker.get_epsolon())
        if current_episode.episode_number % SAVE_STATS_EVERY == 0:
            if False:#blue_decision_maker.type()== AgentType.DQN_keras or blue_decision_maker.type() == AgentType.DQN_basic:
                blue_decision_maker._decision_maker.print_model(initial_state_blue, episode, env.save_folder_path)


    env.end_run()
    if False: #blue_decision_maker.type() == AgentType.DQN_keras or blue_decision_maker.type() == AgentType.DQN_basic:
        blue_decision_maker._decision_maker.print_model(initial_state_blue, episode, env.save_folder_path)

