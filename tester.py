import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.dqn import DQNAgent
from algorithms.reinforce import ReinforceAgent
from algorithms.pid_controller import PID_controller
from networks.value_networks import Qnet
from networks.policy import DiscretePolicy
from utils.eval_utils import plot_training_results, eval_agent

import matplotlib.pyplot as plt

if __name__ == "__main__":
    import numpy as np
    ENV_NAME = 'MountainCarContinuous-v0'
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, "./", episode_trigger=lambda x: True
    )
    print(env.unwrapped.goal_position)

    controller = PID_controller(
        Kp = -5,
        Ki = 0.403,
        Kd =-5
    )

    done = False
    state, info = env.reset(seed=123)
    reward = 0
    rewards = []
    actions = []
    errors = []
    integrals = []
    derivatives = []

    while not done:
        action, error, integral, derivative = controller.update(env.unwrapped.goal_position, state[-1], 1)
        # breakpoint()
        action = np.clip(action, -1.0, 1.0)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        # breakpoint()
        rewards.append(reward)
        actions.append(action)
        errors.append(error)
        integrals.append(integral)
        derivatives.append(derivative)

        done = terminated or truncated
        state = next_state
    env.close()
    fig, ax = plt.subplots(2, 3, figsize = (10, 7), sharex=True)
    ax = ax.flatten()
    steps = len(rewards)
    ax[0].plot(range(steps), rewards, label = 'Temperature')
    ax[0].axhline(0, color='r', linestyle='--', label="Target Temp")
    ax[0].set_ylabel("Reward")
    ax[0].set_title("PID-Cart")
    ax[0].set_xlabel("Time Steps")
    ax[0].grid()

    ax[1].plot(range(steps), actions, label = 'control signals')
    ax[1].set_ylabel("Control Signals")
    ax[1].set_xlabel("Time Steps")
    ax[1].grid()

    ax[2].plot(range(steps), errors, label = 'errors')
    ax[2].set_ylabel("Errors")
    ax[2].set_xlabel("Time Steps")
    ax[2].grid()

    ax[3].plot(range(steps), integrals, label='integrals')
    ax[3].set_ylabel("Integrals")
    ax[3].set_xlabel("Time Steps")
    ax[3].grid()

    ax[4].plot(range(steps), derivatives, label = 'derivatives')
    ax[4].set_ylabel("Derivatives")
    ax[4].set_xlabel("Time Steps")
    ax[4].grid()
    
    # plt.title("PID-Controlled Thermostat")
    # plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.show()

    # q_net = Qnet(
    #     env.observation_space.shape[0],
    #     env.action_space.n,
    #     128,
    #     2
    # )
    # policy = DiscretePolicy(
    #     env.observation_space.shape[0],
    #     env.action_space.n,
    #     10,

    # )

    # agent = DQNAgent(
    #     q_net, 
    #     lr=1.0e-4, 
    #     optimiser=optim.AdamW, 
    #     buffer_size=10000,
    #     mini_batch_size=128,
    #     epsilon=.9,
    #     discount_rate=.99,
    #     num_eval_episodes=10)
    # agent = ReinforceAgent(policy, policy_lr=1.0e-2, optimiser=optim.SGD)

    # agent.train(
    #     env_name = ENV_NAME, 
    #     num_envs=6,
    #     num_iters=250, 
    #     steps_per_update=501, 
    #     eval_freq=50)
    # env_name:str, num_envs:int, num_iters: int, steps_per_update:int, eval_freq: int

    # agent.train(env, num_iters=1000, batch_size=1)
    # agent.train(ENV_NAME, num_envs=4, num_iters=31, steps_per_iter=500, eval_freq=10)
    # env_name: str, num_envs: int, num_iters:int, steps_per_iter:int, eval_freq: int

    # print(agent.eval_results)
    # plot_training_results(agent.batch_results)
    # test_env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    # eval_agent(agent, test_env, 3, save_video=True)
