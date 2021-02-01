
from matplotlib import style

from Arena.CState import State
from Arena.Entity import Entity
from RafaelPlayer.Qtable_DecisionMaker import Qtable_DecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import *
from tqdm import tqdm
import argparse
from DQN.deeprl_prj.dqn_keras import DQNAgent, save_scalar
from DQN.deeprl_prj.core import Sample

style.use("ggplot")

def get_args():
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Seaquest-v0', help='Atari env name')
    parser.add_argument('-o', '--output', default='./log/', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--batch_size', default=32, type=int, help='Minibatch size')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--initial_epsilon', default=1.0, type=float, help='Initial exploration probability in epsilon-greedy')
    parser.add_argument('--final_epsilon', default=0.05, type=float, help='Final exploration probability in epsilon-greedy')
    parser.add_argument('--exploration_steps', default=1000000, type=int, help='Number of steps over which the initial value of epsilon is linearly annealed to its final value')
    parser.add_argument('--num_samples', default=100000000, type=int, help='Number of training samples from the environment in training')
    parser.add_argument('--num_frames', default=4, type=int, help='Number of frames to feed to Q-Network')
    parser.add_argument('--frame_width', default=SIZE_X, type=int, help='Resized frame width')
    parser.add_argument('--frame_height', default=SIZE_X, type=int, help='Resized frame height')
    parser.add_argument('--replay_memory_size', default=1000000, type=int, help='Number of replay memory the agent uses for training')
    parser.add_argument('--target_update_freq', default=10000, type=int, help='The frequency with which the target network is updated')
    parser.add_argument('--train_freq', default=4, type=int, help='The frequency of actions wrt Q-network update')
    parser.add_argument('--save_freq', default=50000, type=int, help='The frequency with which the network is saved')
    parser.add_argument('--eval_freq', default=50000, type=int, help='The frequency with which the policy is evlauted')
    parser.add_argument('--num_burn_in', default=50000, type=int, help='Number of steps to populate the replay memory before training starts')
    parser.add_argument('--load_network', default=False, action='store_true', help='Load trained mode')
    parser.add_argument('--load_network_path', default='', help='the path to the trained mode file')
    parser.add_argument('--net_mode', default='dqn', help='choose the mode of net, can be linear, dqn, duel')
    parser.add_argument('--max_episode_length', default = 10000, type=int, help = 'max length of each episode')
    parser.add_argument('--num_episodes_at_test', default = 20, type=int, help='Number of episodes the agent plays at test')
    parser.add_argument('--ddqn', default=False, dest='ddqn', action='store_true', help='enable ddqn')
    parser.add_argument('--train', default=True, dest='train', action='store_true', help='Train mode')
    parser.add_argument('--test', dest='train', action='store_false', help='Test mode')
    parser.add_argument('--no_experience', default=False, action='store_true', help='do not use experience replay')
    parser.add_argument('--no_target', default=False, action='store_true', help='do not use target fixing')
    parser.add_argument('--no_monitor', default=False, action='store_true', help='do not record video')
    parser.add_argument('--task_name', default='', help='task name')
    parser.add_argument('--recurrent', default=False, dest='recurrent', action='store_true', help='enable recurrent DQN')
    parser.add_argument('--a_t', default=False, dest='a_t', action='store_true', help='enable temporal/spatial attention')
    parser.add_argument('--global_a_t', default=False, dest='global_a_t', action='store_true', help='enable global temporal attention')
    parser.add_argument('--selector', default=False, dest='selector', action='store_true', help='enable selector for spatial attention')
    parser.add_argument('--bidir', default=False, dest='bidir', action='store_true', help='enable two layer bidirectional lstm')

    args = parser.parse_args()
    # args.output = get_output_folder(args, args.output, args.env, args.task_name)

    # env = gym.make(args.env)
    # print("==== Output saved to: ", args.output)
    # print("==== Args used:")
    # print(args)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.

    num_actions = NUMBER_OF_ACTIONS
    print(">>>> Game ", args.env, " #actions: ", num_actions)
    return args, num_actions


# MAIN:
if __name__ == '__main__':

    env = Environment()

    # blue_decision_maker = RafaelDecisionMaker(EASY_AGENT)
    # red_decision_maker = RafaelDecisionMaker(EASY_AGENT)

    blue_decision_maker = Qtable_DecisionMaker()
    # args, num_actions = get_args()
    # blue_decision_maker = DQNAgent(args, num_actions)
    red_decision_maker = Qtable_DecisionMaker()

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

        steps_current_game = 0
        idx_episode = 1
        episode_loss = .0
        episode_frames = 0
        episode_reward = .0
        episode_raw_reward = .0
        episode_target_value = .0
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
            if blue_decision_maker.type() == AgentType.DQN_basic:
                burn_in = True
                action_state = blue_decision_maker.history_processor.process_state_for_network(
                    blue_decision_maker.atari_processor.process_state_for_network(observation_for_blue.img))
                if number_of_frames < 50000: #Number of steps to populate the replay memory before training starts
                    policy_type = "UniformRandomPolicy"
                else:
                    policy_type = "LinearDecayGreedyEpsilonPolicy"
                action = blue_decision_maker.select_action(action_state, True, policy_type=policy_type)
                processed_state = blue_decision_maker.atari_processor.process_state_for_memory(observation_for_blue.img)
                env.blue_player.action(action)  # take the action!

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
            if blue_decision_maker.type() == AgentType.DQN_basic:
                # update Q-network
                processed_next_state = blue_decision_maker.atari_processor.process_state_for_network(new_observation_for_blue.img)
                action_next_state = np.dstack((action_state, processed_next_state))
                action_next_state = action_next_state[:, :, 1:]
                processed_reward = blue_decision_maker.atari_processor.process_reward(reward_step_blue)
                blue_decision_maker.memory.append(processed_state, action, processed_reward, current_episode.is_terminal)
                current_sample = Sample(action_state, action, processed_reward, action_next_state, current_episode.is_terminal)
                episode_frames += 1
                episode_reward += processed_reward
                episode_raw_reward += reward_step_blue

            # update Q-table
            red_decision_maker.update_context(new_observation_for_red,
                                              reward_step_red,
                                              current_episode.is_terminal)

            current_episode.print_episode(env, steps_current_game)
            if current_episode.is_terminal:
                env.update_win_counters()

                if blue_decision_maker.type() == AgentType.DQN_basic:
                    #for DQN agent
                    last_frame = blue_decision_maker.atari_processor.process_state_for_memory(observation_for_blue.img)
                    # action, reward, done doesn't matter here
                    blue_decision_maker.memory.append(last_frame, action, 0, current_episode.is_terminal)
                    if not number_of_frames < 50000:
                        if number_of_frames % blue_decision_maker.train_freq == 0:
                            loss, target_value = blue_decision_maker.update_policy(current_sample)
                            episode_loss += loss
                            episode_target_value += target_value
                        # update freq is based on train_freq
                        if number_of_frames % (blue_decision_maker.train_freq * blue_decision_maker.target_update_freq) == 0:
                            # target updates can have the option to be hard or soft
                            # related functions are defined in deeprl_prj.utils
                            # here we use hard target update as default
                            blue_decision_maker.target_network.set_weights(blue_decision_maker.q_network.get_weights())
                        if number_of_frames % blue_decision_maker.save_freq == 0:
                            blue_decision_maker.save_model(episode)
                        if number_of_frames % (blue_decision_maker.eval_freq * blue_decision_maker.train_freq) == 0:
                            episode_reward_mean, episode_reward_std, eval_count = blue_decision_maker.evaluate(env, 20, eval_count,
                                                                                                MAX_STEPS_PER_EPISODE, True)
                            save_scalar(number_of_frames, 'eval/eval_episode_reward_mean', episode_reward_mean, blue_decision_maker.writer)
                            save_scalar(number_of_frames, 'eval/eval_episode_reward_std', episode_reward_std, blue_decision_maker.writer)


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

