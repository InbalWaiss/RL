
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

style.use("ggplot")

# MAIN:
if __name__ == '__main__':

    env = Environment()

    # blue_decision_maker = RafaelDecisionMaker(EASY_AGENT)
    # red_decision_maker = RafaelDecisionMaker(EASY_AGENT)

    # blue_decision_maker = RafaelDecisionMaker()
    # args, num_actions = get_args()
    # blue_decision_maker = DQNAgent(args, num_actions)
    red_decision_maker = RafaelDecisionMaker()
    blue_decision_maker = DQNAgent.DQNAgent()

    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)

    number_of_frames = 0

    for episode in tqdm(range(1, NUM_OF_EPISODES + 1), ascii=True, unit='episodes'):

        current_episode = Episode(episode)

        # set new start position for the players
        env.blue_player._choose_random_position()
        env.red_player._choose_random_position()

        # get observation
        initial_state_blue: State = env.get_observation_for_blue()
        initial_state_red: State = env.get_observation_for_red()

        # initialize the decision_makers for the players
        if blue_decision_maker.type() == AgentType.Q_table:
            blue_decision_maker.set_initial_state(initial_state_blue)
        if red_decision_maker.type() == AgentType.Q_table:
            red_decision_maker.set_initial_state(initial_state_red)

        # initialize the decision_makers for the players
        if blue_decision_maker.type() == AgentType.DQN:
            blue_decision_maker._decision_maker.tensorboard.step = episode
        if red_decision_maker.type() == AgentType.Q_table:
            red_decision_maker.set_initial_state(initial_state_red)

        for steps_current_game in range(1, MAX_STEPS_PER_EPISODE + 1):

            number_of_frames += 1
            env.number_of_steps += 1

            # get observation
            observation_for_blue: State = env.get_observation_for_blue()
            observation_for_red: State = env.get_observation_for_red()

            # check of the start state is terminal
            current_episode.is_terminal = env.check_terminal()
            if current_episode.is_terminal:
                env.tie_count+=1
                env.starts_at_win += 1
                current_episode.episode_reward_blue = 0
                break

            # blue DQN #
            if blue_decision_maker.type() == AgentType.DQN:
                action_blue = blue_decision_maker._decision_maker._get_action(observation_for_blue.img)
                # action = blue_decision_maker.select_action()
                env.blue_player.action(action_blue)  # take the action!

            if blue_decision_maker.type() == AgentType.Q_table:
                ##### Blue's turn! #####
                action: AgentAction = blue_decision_maker.get_action(observation_for_blue)
                env.blue_player.action(action)  # take the action!

            ##### Red's turn! #####
            action: AgentAction = red_decision_maker.get_action(observation_for_red)
            env.red_player.action(action) # take the action!

            # get new observation
            new_observation_for_blue: State = env.get_observation_for_blue()
            new_observation_for_red: State = env.get_observation_for_red()

            # handle reward
            reward_step_blue, reward_step_red = env.handle_reward()
            current_episode.episode_reward_blue = reward_step_blue

            # check terminal
            current_episode.is_terminal = env.check_terminal()

            if blue_decision_maker.type() == AgentType.Q_table:
                # update Q-table blue
                blue_decision_maker.update_context(new_observation_for_blue,
                                                   reward_step_blue,
                                                   current_episode.is_terminal)

            # blue DQN #
            if blue_decision_maker.type() == AgentType.DQN:
                blue_decision_maker._decision_maker.update_replay_memory((observation_for_blue.img, action_blue, reward_step_blue, new_observation_for_blue.img, current_episode.is_terminal))
                blue_decision_maker._decision_maker.train(current_episode.is_terminal, episode)


            # update Q-table
            red_decision_maker.update_context(new_observation_for_red,
                                              reward_step_red,
                                              current_episode.is_terminal)

            current_episode.print_episode(env, steps_current_game)
            if current_episode.is_terminal:
                env.update_win_counters()

                # if blue_decision_maker.type() == AgentType.DQN:
                #     #for DQN agent
                #     (state, action, reward, new_state, is_terminal)
                #     last_frame = observation_for_blue.img #blue_decision_maker.atari_processor.process_state_for_memory(observation_for_blue.img)
                #     # action, reward, done doesn't matter here
                #     blue_decision_maker.replay_memory.append(new_observation_for_blue, action, reward_step_blue, last_frame, current_episode.is_terminal)
                #

                break

        if steps_current_game == MAX_STEPS_PER_EPISODE:
            # if we exited the loop because we reached MAX_STEPS_PER_EPISODE
            env.update_win_counters()
            current_episode.is_terminal = True


        # for statistics
        env.episodes_rewards.append(current_episode.episode_reward_blue)
        env.steps_per_episode.append(steps_current_game)

        # print info of episode:
        current_episode.print_info_of_episode(env, steps_current_game)


    env.end_run()

