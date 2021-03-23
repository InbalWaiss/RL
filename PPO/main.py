from ppo_torch import Agent
from Arena.Entity import Entity
# from utils import plot_learning_curve
from Arena.Environment import Environment, Episode
from Common.constants import *
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':
    env = Environment()
    env.blue_player = Entity()
    env.red_player = Entity()
    # env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = Agent(n_actions=NUMBER_OF_ACTIONS, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=(3,15,15))
    n_games = 300

    figure_file = 'plots/combat.png'

    best_score = -500# env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in tqdm(range(1, n_games + 1), ascii=True, unit='episodes'):
    # for i in range(n_games):

        current_episode = Episode(i, show_always=True)

        # set new start position for the players
        env.reset_players_positions(i)

        state = env.get_observation_for_blue()
        done = False
        score = 0
        steps_current_game = 0
        while not done:
            # action, prob, val = agent.choose_action(observation)
            action, prediction = agent.act(state)
            steps_current_game+=1
            next_state = env.get_observation_for_blue()

            done = (env.compute_terminal() is not WinEnum.NoWin)
            reward, _ = env.handle_reward(steps_current_game)
            # observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            prob =0
            val = 0
            agent.remember(state, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
