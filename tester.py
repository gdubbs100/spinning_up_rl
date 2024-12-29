import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.dqn import DQNAgent
from algorithms.reinforce import ReinforceAgent
from networks.value_networks import Qnet
from networks.policy import DiscretePolicy
from utils.eval_utils import plot_training_results, eval_agent

if __name__ == "__main__":
    ENV_NAME = 'Acrobot-v1'
    env = gym.make(ENV_NAME)
    q_net = Qnet(
        env.observation_space.shape[0],
        env.action_space.n,
        128,
        2
    )
    # policy = DiscretePolicy(
    #     env.observation_space.shape[0],
    #     env.action_space.n,
    #     10,

    # )

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
    # agent = ReinforceAgent(policy, policy_lr=1.0e-2, optimiser=optim.SGD)

    # agent.train(
    #     env_name = ENV_NAME, 
    #     num_envs=6,
    #     num_iters=250, 
    #     steps_per_update=501, 
    #     eval_freq=50)
    # env_name:str, num_envs:int, num_iters: int, steps_per_update:int, eval_freq: int

    # agent.train(env, num_iters=1000, batch_size=1)
    agent.train(ENV_NAME, num_episodes=50, eval_freq=10)

    print(agent.eval_results)
    plot_training_results(agent.batch_results)
    # test_env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    # eval_agent(agent, test_env, 3, save_video=True)
