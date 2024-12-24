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
    # TODO:
    # 1. change action selection - use max Q value
    # 2. create target network
    # 3. need a replay buffer, w batch sampling and shuffling
    # 4. loss is calculated using MSE over value
    # 5. add epsilon value for exploration

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
        self.mini_batch_size = mini_batch_size

        self.buffer = ReplayBuffer(max_size=buffer_size, with_replacement=True)
 
        self.optimiser = optimiser(
            self.q_network.parameters(), 
            lr=lr) 

        ## TODO: convert this to tensorboard logger?
        self.batch_results = dict()

        ## setup 
        self.init_records()

    def init_records(self):
        self.records = {
            'state':[],
            'action':[],
            'Q': [],
            'next_Q': [],
            'next_state':[],
            'reward': [],
            'done':[]
        }

    # def append_record(
    #     self, 
    #     state, 
    #     action,  
    #     next_state, 
    #     # Q,
    #     # next_Q,
    #     reward, 
    #     done
    #     ):
    #     self.records['state'].append(state)
    #     self.records['action'].append(action)
    #     self.records['next_state'].append(next_state)
    #     # self.records['Q'].append(Q)
    #     # self.records['next_Q'].append(next_Q)
    #     self.records['reward'].append(reward)
    #     self.records['done'].append(done)

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

    def train(self, env, num_iters, batch_size):
        
        for episode in range(num_iters):
            self.optimiser.zero_grad()
            self.sample_trajectory(env, batch_size)

            ## do the update
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

                loss = torch.pow(
                    rewards + self.discount_rate * next_values * (1 - dones) - values,
                    2
                ).mean()


                loss.backward()
                self.optimiser.step()

                ## log for post training review
                self.log_batch_results(
                    batch=episode,
                    q_values=values.mean().detach().numpy() / batch_size,
                    next_q_values=next_values.mean().numpy() / batch_size,
                    rewards=rewards.sum(),
                    loss=loss.detach().numpy(),
                    episode_len = rewards.size(-1) / batch_size
                )

                # TODO: update target network every k iters
                if episode % 5 == 0:
                    # self.target_network = copy.deepcopy(
                    #     self.q_network
                    # )
                    target_net_state_dict = self.target_network.state_dict()
                    policy_net_state_dict = self.q_network.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*.05 + target_net_state_dict[key]*(1-.05)
                    self.target_network.load_state_dict(target_net_state_dict)

            ## clear out records for next batch
            self.init_records()

    def act(self, x):
        Q = self.q_network(x)
        if torch.rand(1) < self.epsilon:
            # uniform sampling
            action = dist.Categorical(
                logits=torch.ones_like(Q)
                ).sample()
        else:
            action = torch.argmax(Q, dim=-1)
        return action, Q.max(dim=-1)[0]

    def sample_trajectory(self, env: gym.Env, n_samples:int):

        for episode in range(n_samples):
            state, info = env.reset()
            done = False
            while not done:
                
                with torch.no_grad():
                    action, _ = self.act(state) 

                next_state, reward, terminated, truncated, info = env.step(action.numpy())

                t= Transition(
                    action=action.numpy(), 
                    state=state, 
                    reward=reward, 
                    done=done, 
                    next_state=next_state
                )

                self.buffer.insert(t)

                # with torch.no_grad():
                #     next_Q = self.target_network(next_state).max(dim=-1)[0]
                done = terminated or truncated
                
                # self.append_record(
                #     state, 
                #     action, 
                #     next_state, 
                #     Q,
                #     next_Q,
                #     reward, 
                #     done
                # )

                state = next_state