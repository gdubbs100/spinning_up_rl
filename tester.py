import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.dqn import DQN
from networks.policy import DiscretePolicy
from utils.eval_utils import plot_training_results, eval_agent
class Qnet(nn.Module):

    def __init__(self, state_dim, action_dim, intermediate_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, action_dim)
        )
    
    def forward(self, x):
        x = torch.tensor(x)
        x = self.net(x)
        return x

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    q_net = Qnet(
        env.observation_space.shape[0],
        env.action_space.n,
        128
    )

    agent = DQN(
        q_net, 
        lr=1.0e-4, 
        optimiser=optim.AdamW, 
        buffer_size=10000,
        mini_batch_size=128,
        epsilon=.9,
        discount_rate=.99)

    # agent.train(env, num_iters=1000, batch_size=1)
    agent.train(env, n_samples=600)
    # breakpoint()
    plot_training_results(agent.batch_results)
    # test_env = gym.make('CartPole-v1', render_mode = 'rgb_array')
    # eval_agent(agent, test_env, 3, save_video=True)
