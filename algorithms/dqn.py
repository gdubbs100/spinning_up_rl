import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import Dataset, RandomSampler

import gymnasium as gym

from utils.training_utils import calc_returns

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

        



class DQNAgent:

    def __init__(
        self, 
        q_network: nn.Module, 
        optimiser: optim.Optimizer, 
        buffer_size: int = 1000,
        mini_batch_size: int = 32,
        discount_rate: float=.99, 
        lr: float=.01,
        epsilon: float = .1,
        eps_end: float = .1,
        eps_decay: int = 1000,
        tau: float = .005,
        target_update_freq: int = 10,
        eval_freq: int = 10,
        num_eval_episodes: int = 10,
        double_dqn: bool=True
        ):

        self.q_network = q_network
        self.target_network = copy.deepcopy(self.q_network)
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.eps_start = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done=0
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.mini_batch_size = mini_batch_size
        self.double_dqn = double_dqn

        self.buffer = ReplayBuffer(max_size=buffer_size)
 
        self.optimiser = optimiser(
            self.q_network.parameters(), 
            lr=lr) 

        ## TODO: convert this to tensorboard logger?
        self.batch_results = dict()

        ## evaluation
        self.num_eval_episodes = num_eval_episodes
        self.eval_results = dict()

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
                (
                    actions,
                    states,
                    rewards,
                    dones,
                    next_states
                ) = self.buffer.sample(self.mini_batch_size)

                # create computation graph
                values = self.q_network(states)[torch.arange(self.mini_batch_size), actions]
                # values = values[torch.arange(mini_batch_size), actions]

                if self.double_dqn:
                    with torch.no_grad():
                        
                        next_actions =  torch.argmax(self.q_network(next_states), dim=-1)
                        # breakpoint()
                        next_values = self.target_network(next_states)[torch.arange(self.mini_batch_size), next_actions]

                else:
                    with torch.no_grad():
                        next_values = self.target_network(next_states).max(-1).values

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

    def act(self, x):

        self.increment_epsilon()
        Q = self.q_network(x)
        if torch.rand(1) < self.epsilon:
            # uniform sampling
            action = dist.Categorical(
                logits=torch.ones_like(Q)
                ).sample()
        else:
            action = torch.argmax(Q, dim=-1)
        return action, Q.max(dim=-1)[0]

    def increment_epsilon(self):
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
    
    def update_target_network(self):
        target_net_state_dict = self.target_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                policy_net_state_dict[key]*self.tau + 
                target_net_state_dict[key]*(1-self.tau)
            )
        self.target_network.load_state_dict(target_net_state_dict)
    
    def evaluate(self, env_name: str, num_eval_episodes):
        rewards = []
        env = gym.make(env_name)
        for episode in range(num_eval_episodes):
            state, info = env.reset()
            done = False
            
            while not done:
                
                with torch.no_grad():
                    action, _ = self.act(state) 

                next_state, reward, terminated, truncated, info = env.step(action.numpy())
                done = terminated or truncated

                rewards.append(reward)

                state = next_state
        return sum(rewards) / num_eval_episodes

    def train(self, env_name: str, num_episodes:int, eval_freq: int):
        env = gym.make(env_name)

        for episode in range(num_episodes):
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

                ## TODO: is this the right place to log?
                self.log_batch_results(
                    batch=episode,
                    q_values = values.detach().mean().numpy(),
                    next_q_values = next_values.mean().numpy(),
                    rewards=sum(rewards),
                    loss=loss.detach().mean().numpy(),
                    episode_len = len(rewards)
                )

                self.update_target_network()
                
                state = next_state

                if done:

                    # if episode % self.target_update_freq == 0:
                        
                    if episode % eval_freq == 0:
                        print(f"evaluating at {episode}...")
                        self.eval_results[episode] = (
                            self.evaluate(env_name, self.num_eval_episodes)
                        )



