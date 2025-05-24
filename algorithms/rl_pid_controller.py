import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import gymnasium as gym
import numpy as np

from pid_controller import PID_controller


class PID_CONTROLLER_AGENT:

    def __init__(
            self, 
            state_dim, 
            action_dim,
            optimiser: optim.Optimizer, 
            lr: float=.01,
        ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU()
        ).to(device=self.device)
        ## parameterise the pid controller
        self.param_net = nn.Sequential(
            nn.Linear(state_dim, 3*action_dim)
        ).to(device=self.device)
        ## set the rewards for each action
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, action_dim),
            nn.ReLU() ## always positive
        ).to(device=self.device)
        self.seed=1234
        ## optimiser
        self.optimiser = optimiser(lr=lr)
    
    def get_pid_values(self, inputs):
        x = torch.Tensor(inputs).to(device=self.device)
        x = self.net(x)
        params = self.param_net(x)
        targets = self.reward_net(x)
        return params, targets
        
    def train(self, env: gym.Env, num_iters:int):
        state, info = env.reset(seed=self.seed)
        reward = 0
        for step in range(num_iters):
            ## while not done?
            params, target = self.get_pid_values(state)
            controller = PID_controller(*params.cpu().detach().numpy())
            action, _, _, _ = controller.update(target.cpu().detach().numpy(), reward, 1)
            next_state, reward, terminated, truncated, info = env.step(
                action[0]
            )
            print(action, state, next_state)
            done = terminated or truncated
            ## update
            state = next_state
        
        def update(self):
            pass


if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    agent = PID_CONTROLLER_AGENT(
        state_dim= env.observation_space.shape[0],
        action_dim = env.action_space.shape[0],
    )

    agent.train(env, 1)