
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

    red_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker()
    # red_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker(UPDATE_CONTEXT=False, path_model_to_load='qtable_red-1000000.pickle')
    # red_decision_maker = Qtable_DecisionMaker('keep_training_qtable_1900000_DQNkeras_900000\qtable_red-900000.pickle')
    # red_decision_maker = DQNAgent.DQNAgent()
    # red_decision_maker = DQNAgent_keras.DQNAgent_keras()
    # red_decision_maker = DQNAgent_temporalAttention.DQNAgent_temporalAttention()


    # blue_decision_maker = Qtable_DecisionMaker('qtable_blue-1000000_old_terminal_state.pickle')
    # blue_decision_maker = Qtable_DecisionMaker.Qtable_DecisionMaker()
    # blue_decision_maker = DQNAgent_keras.DQNAgent_keras()
    # blue_decision_maker = DQNAgent_keras.DQNAgent_keras(UPDATE_CONTEXT=False, path_model_to_load='32X64X64X512X9_blue_70001_ 249.00max_ -97.22avg_-249.00min__1613486704.model')
    blue_decision_maker = DQNAgent_spatioalAttention.DQNAgent_spatioalAttention(UPDATE_CONTEXT=False, path_model_to_load='statistics/17_02_06_09_DQNAgent_spatioalAttention_Q_table/qnet840000.cptk')
    # blue_decision_maker = DQNAgent_keras.DQNAgent_keras('DQN_keras_blue_32X64X64X512X9_200001_ 249.00max_-131.98avg_-249.00min__1612271297.model')
    # blue_decision_maker = DQNAgent_temporalAttention.DQNAgent_temporalAttention(UPDATE_BLUE_CONTEXT)#
    # blue_decision_maker = DQNAgent.DQNAgent(path_model_to_load='DQN_basic_20000.model')


    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)

    print_start_of_game_info(blue_decision_maker, red_decision_maker)

    NUM_OF_EPISODES = env.NUMBER_OF_EPISODES
    for episode in tqdm(range(1, NUM_OF_EPISODES + 1), ascii=True, unit='episodes'):

        current_episode = Episode(episode, show_always=False if IS_TRAINING else True)

        # set new start position for the players
        env.reset_players_positions(episode)

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
            observation_for_red: State = env.get_observation_for_red()

            # Check if the start state is terminal
            current_episode.is_terminal = (env.compute_terminal() is not WinEnum.NoWin)
            if current_episode.is_terminal:
                env.tie_count+=1
                env.starts_at_win += 1
                env.starts_at_win_in_last_SHOW_EVERY_games +=1
                current_episode.episode_reward_blue = 0
                break

            ##### Blue's turn! #####
            action_blue: AgentAction = blue_decision_maker.get_action(observation_for_blue)
            env.blue_player.action(action_blue)  # take the action!

            ##### Red's turn! #####
            action_red: AgentAction = red_decision_maker.get_action(observation_for_red)
            env.red_player.action(action_red) # take the action!

            # Get new observations
            new_observation_for_blue: State = env.get_observation_for_blue()
            new_observation_for_red: State = env.get_observation_for_red()

            # Check if terminal
            current_episode.is_terminal = (env.compute_terminal() is not WinEnum.NoWin)

            # Handle rewards
            reward_step_blue, reward_step_red = env.handle_reward(steps_current_game)
            current_episode.episode_reward_blue = reward_step_blue

            # Update models
            blue_decision_maker.update_context(new_observation_for_blue,
                                                  reward_step_blue,
                                                  current_episode.is_terminal)

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
        env.episodes_rewards.append(current_episode.episode_reward_blue)
        env.steps_per_episode.append(steps_current_game)

        # print info of episode:
        current_episode.print_info_of_episode(env, steps_current_game, blue_decision_maker.get_epsolon())


    env.end_run()

