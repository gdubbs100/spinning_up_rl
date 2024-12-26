import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.dqn import DQNAgent
from networks.value_networks import Qnet
from utils.eval_utils import plot_training_results, eval_agent

if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    q_net = Qnet(
        env.observation_space.shape[0],
        env.action_space.n,
        128,
        2
    )

    agent = DQNAgent(
        q_net, 
        lr=1.0e-4, 
        optimiser=optim.AdamW, 
        buffer_size=10000,
        mini_batch_size=128,
        epsilon=.9,
        discount_rate=.99,
        num_eval_episodes=5,
        eval_freq = 10)

    # agent.train(env, num_iters=1000, batch_size=1)
    agent.train("Acrobot-v1", num_episodes=50)
    breakpoint()
    plot_training_results(agent.batch_results)
    # test_env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    # eval_agent(agent, test_env, 3, save_video=True)
