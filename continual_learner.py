import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt

from algorithms.dqn import DQNAgent
from networks.value_networks import Qnet
from environments.continual_environment import ContinualEnv
from utils.continual_training_utils import ContinualTrainer

import gymnasium as gym

if __name__ == "__main__":

    PARAMS = [ 
        {
            "id":"LunarLander-v3",
            "continuous":False,
            "gravity":-10.0,
            "enable_wind": False,
            "wind_power":15.0,
            "turbulence_power":1.5
        },
        # {
        #     "id":"LunarLander-v3",
        #     "continuous":False,
        #     "gravity":-10.0,
        #     "enable_wind": True,
        #     "wind_power":15.0,
        #     "turbulence_power":1.5
        # }
    ]

    tmp_env = gym.make(PARAMS[0]["id"])

    q_net = Qnet(
        tmp_env.observation_space.shape[0],
        tmp_env.action_space.n,
        intermediate_dim=128,
        n_layers=2
    )

    agent = DQNAgent(
        q_net, 
        lr=1.0e-4, 
        optimiser=optim.Adam, 
        buffer_size=1e6,
        mini_batch_size=64,
        epsilon=.9,
        discount_rate=.99,
        num_eval_episodes=10
    )
    
    
    env = ContinualEnv(PARAMS, steps_per_env=2*500000, num_parallel_envs=4)
    trainer = ContinualTrainer(
        env=env, 
        agent=agent
    )
    # breakpoint()
    df = trainer.train(
        steps_per_update=5,
    )
    print(df.tail(10))
    sns.lineplot(
        data=df, 
        x='epoch', 
        y='rewards', 
        hue='curr_task', 
        marker='o'
    )
    plt.show()