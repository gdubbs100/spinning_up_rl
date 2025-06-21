import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from algorithms.dqn import DQNAgent
from networks.value_networks import Qnet
from environments.continual_environment import ContinualEnv
from utils.continual_training_utils import ContinualTrainer

import gymnasium as gym
# import minigrid
# from minigrid.wrappers import ImgObsWrapper

if __name__ == "__main__":

    NUM_PARALLEL_ENVS = 1

    PARAMS = [ 
        # {
        #     "id":"Acrobot-v1",
        #     # "continuous":False,
        #     # "gravity":-10.0,
        #     # "enable_wind": False,
        #     # "wind_power":15.0,
        #     # "turbulence_power":1.5
        # },
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
        #     "wind_power":40.0,
        #     "turbulence_power":2.0
        # }
    ]

    tmp_env = gym.make(PARAMS[0]["id"])

    # Load Q-network from file
    q_net = Qnet(
        tmp_env.observation_space.shape[0],
        tmp_env.action_space.n,
        intermediate_dim=128,
        n_layers=1
    )
    # torch.load("logs/continual_learner/LunarLander-v3_06-08-2025_111432/network.pt")

    agent = DQNAgent(
        q_net, 
        lr=1.0e-4, 
        optimiser=optim.Adam, 
        buffer_size=1e5,
        mini_batch_size=64,# should this be multiplied by num_parallel_envs?
        epsilon=.99,
        eps_end=0.01,
        eps_decay=1e4,
        discount_rate=.99,
        tau=5e-3,
        num_eval_episodes=5
    )
    
    
    env = ContinualEnv(PARAMS, steps_per_env=120000, num_parallel_envs=NUM_PARALLEL_ENVS)
    trainer = ContinualTrainer(
        env=env, 
        agent=agent,
        log_dir=f"./runs/continual_learner/{PARAMS[0]['id']}_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}",
    )
    # breakpoint()
    df = trainer.train(
        update_every=1,
        eval_every=5000
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