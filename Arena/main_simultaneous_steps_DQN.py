
from matplotlib import style

from Arena.CState import State
from Arena.Entity import Entity
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import *
from tqdm import tqdm
import argparse
from DQN.deeprl_prj.dqn_keras import DQNAgent, save_scalar
from DQN.deeprl_prj.core import Sample
from DQN import DQNAgent
from DQN import DQNAgent_keras, DQNAgent_temporalAttention

style.use("ggplot")

# MAIN:
if __name__ == '__main__':

    env = Environment()

    # blue_decision_maker = RafaelDecisionMaker(EASY_AGENT)
    # red_decision_maker = RafaelDecisionMaker(EASY_AGENT)

    # blue_decision_maker = RafaelDecisionMaker()
    # blue_decision_maker = DQNAgent(args, num_actions)
    # red_decision_maker = RafaelDecisionMaker('qtable_red-1000000.pickle')
    # blue_decision_maker = DQNAgent.DQNAgent()
    blue_decision_maker = DQNAgent_temporalAttention.DQNAgent_temporalAttention()
    red_decision_maker = DQNAgent_temporalAttention.DQNAgent_temporalAttention()

    # red_decision_maker = RafaelDecisionMaker()

    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)


    for episode in tqdm(range(1, NUM_OF_EPISODES + 1), ascii=True, unit='episodes'):

        current_episode = Episode(episode)

        # set new start position for the players
        env.blue_player._choose_random_position()
        env.red_player._choose_random_position()

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
            current_episode.is_terminal = env.check_terminal()
            if current_episode.is_terminal:
                env.tie_count+=1
                env.starts_at_win += 1
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

            # Handle rewards
            reward_step_blue, reward_step_red = env.handle_reward()
            current_episode.episode_reward_blue = reward_step_blue

            # Check if terminal
            current_episode.is_terminal = env.check_terminal()

            # Update models
            blue_decision_maker.update_context(new_observation_for_red,
                                              reward_step_red,
                                              current_episode.is_terminal)

            red_decision_maker.update_context(new_observation_for_red,
                                              reward_step_red,
                                              current_episode.is_terminal)

            current_episode.print_episode(env, steps_current_game)
            if current_episode.is_terminal:
                env.update_win_counters()

                break

        if steps_current_game == MAX_STEPS_PER_EPISODE:
            # if we exited the loop because we reached MAX_STEPS_PER_EPISODE
            env.update_win_counters()
            current_episode.is_terminal = True


        # for statistics
        env.episodes_rewards.append(current_episode.episode_reward_blue)
        env.steps_per_episode.append(steps_current_game)

        # print info of episode:
        current_episode.print_info_of_episode(env, steps_current_game, blue_decision_maker._decision_maker._epsilon)


    env.end_run()

