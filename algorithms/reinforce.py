import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import gymnasium as gym

from utils.training_utils import calc_returns

class ReinforceAgent:

    def __init__(self, policy: nn.Module, discount_rate: float=.99, policy_lr: float=.01): 
        self.policy = policy
        self.discount_rate = discount_rate

        ## TODO: make optimiser an input to agent
        self.optimiser = optim.SGD(self.policy.parameters(), lr=policy_lr)
        ## TODO: convert this to tensorboard logger?
        self.batch_results = dict()
        ## setup 
        self.init_records()

    def init_records(self):
        self.records = {
            'state':[],
            'action':[],
            'action_log_prob': [],
            'next_state':[],
            'reward': [],
            'done':[]
        }

    def append_record(
        self, 
        state, 
        action, 
        log_prob, 
        next_state, 
        reward, 
        done
        ):
        self.records['state'].append(state)
        self.records['action'].append(action)
        self.records['action_log_prob'].append(log_prob)
        self.records['next_state'].append(next_state)
        self.records['reward'].append(reward)
        self.records['done'].append(done)

    def log_batch_results(
        self, 
        batch, 
        log_probs, 
        rewards, 
        returns, 
        policy_gradient, 
        policy_entropy,
        episode_len
        ):
        self.batch_results[batch] = {
            'log_probs':log_probs,
            'rewards': rewards,
            'returns': returns,
            'policy_gradient': policy_gradient,
            'policy_entropy': policy_entropy,
            'episode_len': episode_len
        }

    def train(self, env, num_iters, batch_size):
        
        for batch in range(num_iters):
            self.optimiser.zero_grad()
            self.sample_trajectory(env, batch_size)

            ## get values to log
            log_probs = torch.stack(self.records['action_log_prob']) 
            dones = torch.tensor(self.records['done'], dtype=torch.int32)
            rewards = torch.tensor(self.records['reward'])
            returns = calc_returns(rewards, dones, self.discount_rate)

            ## TODO: make baseline a function
            ## add a baseline
            baseline= returns.mean()
            policy_gradient = (-log_probs*(returns- baseline)).mean()

            ## backprop
            policy_gradient.backward()
            self.optimiser.step()

            ## log for post training review
            self.log_batch_results(
                batch=batch,
                log_probs=log_probs.mean().detach().numpy(),
                rewards=rewards.sum().numpy() / batch_size,
                returns=returns.mean().numpy(),
                policy_gradient=policy_gradient.detach().numpy(),
                policy_entropy=(torch.exp(log_probs)*log_probs).mean().detach().numpy(),
                episode_len = rewards.size(-1) / batch_size
            )

            ## clear out records for next batch
            self.init_records()

    def act(self, x):
        action_logits = self.policy(x)
        return dist.Categorical(logits=action_logits)

    def sample_trajectory(self, env: gym.Env, n_samples:int):

        for episode in range(n_samples):
            state, info = env.reset()
            done = False
            while not done:
                
                action_dist = self.act(state) 
                action = action_dist.sample()
                action_log_prob = action_dist.log_prob(action)
                next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
                done = terminated or truncated

                self.append_record(
                    state, 
                    action, 
                    action_log_prob,
                    next_state, 
                    reward, 
                    done
                )

                state = next_state