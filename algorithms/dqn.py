import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import Dataset, RandomSampler

import gymnasium as gym

from utils.training_utils import calc_returns


from torch.utils.data import Dataset
import numpy as np
import numpy.random

class Transition:

    def __init__(self, action, state, reward, next_state, done):
        self.action = action
        self.state = state
        self.reward = reward
        self.done = done
        self.next_state = next_state
    
    def __repr__(self):
        return f"""
        action: {self.action},
        \nstate: {self.state},
        \nreward: {self.reward},
        \ndone: {int(self.done)},
        \nnext_state: {self.next_state}
        """

class ReplayBuffer():

    def __init__(
        self, 
        max_size: int, 
        weight_func: callable = lambda x: [1/len(x)]*len(x),
        with_replacement: bool = False
    ):
        self.max_size = max_size
        self.transitions = []
        self.weight_func = weight_func
        self.with_replacement = with_replacement

    def __len__(self):
        return len(self.transitions)
        
    def insert(self, transition: Transition):
        ## requires insertion of single transition
        if len(self.transitions) + 1 < self.max_size:
            self.transitions.append(transition)
        else:
            self.transitions = self.transitions[1:]
            self.transitions.append(transition)
    
    def calculate_weights(self):
        return self.weight_func(self.transitions)

    def sample_transitions(self, num_samples: int):
        transition_weights = self.calculate_weights()
        return np.random.choice(
                self.transitions,
                p=transition_weights,
                size = num_samples,
                replace = self.with_replacement
            ).tolist()
        
    def sample(self, num_samples: int):

        (
            actions,
            states,
            rewards,
            dones,
            next_states
        ) = zip(
                *[(i.action, i.state, i.reward, i.done, i.next_state) 
                for i in self.sample_transitions(num_samples)]
        )
        # breakpoint()
        ## convert to pytorch
        return (
            torch.tensor(np.array(actions)),
            torch.tensor(np.array(states)),
            torch.tensor(rewards),
            torch.tensor(dones, dtype=torch.int32),
            torch.tensor(np.array(next_states))
        )

        



class DQN:

    def __init__(
        self, 
        q_network: nn.Module, 
        optimiser: optim.Optimizer, 
        buffer_size: int = 1000,
        mini_batch_size: int = 32,
        discount_rate: float=.99, 
        lr: float=.01,
        epsilon: float = .1
        ):

        self.q_network = q_network
        self.target_network = copy.deepcopy(self.q_network)
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.eps_start = epsilon
        self.steps_done=0
        self.mini_batch_size = mini_batch_size

        self.buffer = ReplayBuffer(max_size=buffer_size)
 
        self.optimiser = optimiser(
            self.q_network.parameters(), 
            lr=lr,
            amsgrad=True) 

        ## TODO: convert this to tensorboard logger?
        self.batch_results = dict()

    def log_batch_results(
        self, 
        batch, 
        q_values,
        next_q_values, 
        rewards, 
        loss,
        episode_len
        ):
        self.batch_results[batch] = {
            'q_values':q_values,
            'next_q_values':next_q_values,
            'rewards': rewards,
            'loss': loss,
            'episode_len': episode_len
        }
    
    def update_model(self):
            if self.mini_batch_size <= len(self.buffer):
            # for i in range(num_mini_batches):
                (
                    actions,
                    states,
                    rewards,
                    dones,
                    next_states
                ) = self.buffer.sample(self.mini_batch_size)

                # create computation graph
                values = self.q_network(states)
                values = values[torch.arange(values.size(0)), actions]
                with torch.no_grad():
                    next_values = self.target_network(next_states).max(-1).values

                # loss = torch.pow(
                #     rewards + self.discount_rate * next_values * (1 - dones) - values,
                #     2
                # ).mean()
                criterion = nn.SmoothL1Loss()#nn.MSELoss()#
                loss = criterion(
                    values,
                    rewards + self.discount_rate * next_values * (1 - dones)
                )

                self.optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
                self.optimiser.step()
            else:
                loss = torch.tensor([0.])
                values = torch.tensor([0.])
                next_values = torch.tensor([0.])

            return loss, values, next_values



    # def train(self, env, num_iters, batch_size):
        
    #     for episode in range(num_iters):
            
    #         ## TODO: need to move the update into the sampling trajectory!
    #         ## updates happens at each time step
    #         training_rewards = self.sample_trajectory(env, batch_size)

    #         ## do the update
    #         if self.mini_batch_size <= len(self.buffer):
    #         # for i in range(num_mini_batches):
    #             (
    #                 actions,
    #                 states,
    #                 rewards,
    #                 dones,
    #                 next_states
    #             ) = self.buffer.sample(self.mini_batch_size)
                
    #             # create computation graph
    #             values = self.q_network(states)
    #             values = values[torch.arange(values.size(0)), actions]
    #             with torch.no_grad():
    #                 next_values = self.target_network(next_states).max(-1).values

    #             # loss = torch.pow(
    #             #     rewards + self.discount_rate * next_values * (1 - dones) - values,
    #             #     2
    #             # ).mean()
    #             criterion = nn.SmoothL1Loss()#nn.MSELoss()#
    #             loss = criterion(
    #                 values,
    #                 rewards + self.discount_rate * next_values * (1 - dones)
    #             )

    #             self.optimiser.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
    #             self.optimiser.step()

    #             # TODO: update target network every k iters
    #             if episode % 10 == 0:

    #                 target_net_state_dict = self.target_network.state_dict()
    #                 policy_net_state_dict = self.q_network.state_dict()
    #                 for key in policy_net_state_dict:
    #                     target_net_state_dict[key] = policy_net_state_dict[key]*.005 + target_net_state_dict[key]*(1-.005)
    #                 self.target_network.load_state_dict(target_net_state_dict)

    #             self.log_batch_results(
    #                 batch=episode,
    #                 q_values = values.detach().mean().numpy(),
    #                 next_q_values = next_values.mean().numpy(),
    #                 rewards=sum(training_rewards),
    #                 loss=loss.detach().numpy(),
    #                 episode_len = len(training_rewards)
    #             )
    #         else:
    #             self.log_batch_results(
    #                 batch=episode,
    #                 q_values = 0,
    #                 next_q_values = 0,
    #                 rewards=sum(training_rewards),
    #                 loss=0,
    #                 episode_len = len(training_rewards)
    #             )




    def act(self, x):

        self.epsilon = .01 + (self.eps_start - .01) * \
            np.exp(-1. * self.steps_done / 10000)
        self.steps_done += 1
        Q = self.q_network(x)
        if torch.rand(1) < self.epsilon:
            # uniform sampling
            action = dist.Categorical(
                logits=torch.ones_like(Q)
                ).sample()
        else:
            action = torch.argmax(Q, dim=-1)
        return action, Q.max(dim=-1)[0]

    # def sample_trajectory(self, env: gym.Env, n_samples:int):
    def train(self, env: gym.Env, n_samples:int):

        for episode in range(n_samples):
            state, info = env.reset()
            done = False
            rewards = []
            while not done:
                
                with torch.no_grad():
                    action, _ = self.act(state) 

                next_state, reward, terminated, truncated, info = env.step(action.numpy())
                done = terminated or truncated
                rewards.append(reward)
                t= Transition(
                    action=action.numpy(), 
                    state=state, 
                    reward=reward, 
                    done=done, 
                    next_state=next_state
                )

                self.buffer.insert(t)

                loss, values, next_values = self.update_model()

                # TODO: update target network every k iters
                if episode % 10 == 0:

                    target_net_state_dict = self.target_network.state_dict()
                    policy_net_state_dict = self.q_network.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*.005 + target_net_state_dict[key]*(1-.005)
                    self.target_network.load_state_dict(target_net_state_dict)

                self.log_batch_results(
                    batch=episode,
                    q_values = values.detach().mean().numpy(),
                    next_q_values = next_values.mean().numpy(),
                    rewards=sum(rewards),
                    loss=loss.detach().mean().numpy(),
                    episode_len = len(rewards)
                )

                
                state = next_state

                

            # return rewards

