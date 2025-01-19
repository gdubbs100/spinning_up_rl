import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import gymnasium as gym
import numpy as np

from utils.training_utils import calc_returns

## TODO: consider if there is a way to start training from end of unfinished episode
def reset_after_last_zero(mask):
    """
    Filter out unfinished episodes from loss calc
    """
    result = mask.clone()
    for row in range(result.size(0)):
        ## if no episode completes
        if (result[row].size()[0]==result[row].sum().item()):
            result[row, :] = -1
        else:
            last_zero_idx = (
                (result[row] == 0).nonzero(as_tuple=True)[0].max().item() + 1
                if (result[row] == 0).any() else -1
            )

            if last_zero_idx != -1:
                result[row, last_zero_idx:] = -1

    return result

class ReinforceAgent:

    def __init__(
        self, 
        policy: nn.Module, 
        optimiser: optim.Optimizer, 
        discount_rate: float=.99, 
        policy_lr: float=.01,
        num_eval_episodes: int = 5,
        seed: int = 42
        ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.discount_rate = discount_rate
 
        self.optimiser = optimiser(self.policy.parameters(), lr=policy_lr) 

        ## TODO: convert this to tensorboard logger?
        self.batch_results = dict()

        ## evaluate
        self.eval_results = dict()
        self.num_eval_episodes = num_eval_episodes

        ## setup 
        self.init_records()

        self.seed = seed

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

    def evaluate(self, env_name: str, num_eval_episodes:int):
        rewards = []
        env = gym.make(env_name)
        for episode in range(num_eval_episodes):
            state, info = env.reset(seed=self.seed)
            done = False
            
            while not done:
                
                with torch.no_grad():
                    action, _ = self.act(state) 

                next_state, reward, terminated, truncated, info = env.step(
                    action.squeeze(-1).cpu().numpy().item()
                )
                done = terminated or truncated

                rewards.append(reward)

                state = next_state
        return sum(rewards) / num_eval_episodes

    def sample_trajectory(self, env: gym.Env, num_steps:int):
        state, info = env.reset(seed=self.seed)
        for step in range(num_steps):
            action, action_log_prob = self.act(state) 
            next_state, reward, terminated, truncated, info = env.step(
                action.cpu().detach().numpy()
            )
            ## TODO: calculation of returns would benefit from differentiating terminated/ truncated
            ## but need to keep truncated for number of episode calcs.
            done = [any(i) for i in zip(terminated, truncated)] 

            self.append_record(
                state, 
                action, 
                action_log_prob,
                next_state, 
                reward, 
                done
            )

            state = next_state

    def train(self, env_name:str, num_envs:int, num_iters: int, steps_per_iter:int, eval_freq: int):
        env = gym.make_vec(env_name, num_envs)
        for batch in range(num_iters):
            
            self.sample_trajectory(env, steps_per_iter)

            ## get values to log
            log_probs = torch.stack(self.records['action_log_prob']).T.to(self.device) 
            dones = torch.tensor(np.array(self.records['done']), dtype=torch.int32).T.to(self.device)
            rewards = torch.tensor(np.array(self.records['reward'])).T.to(self.device)
            returns = calc_returns(rewards, dones, self.discount_rate)
            ## standardise returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
           
            policy_gradient = (-log_probs*returns)
            mask = reset_after_last_zero(1-dones).flatten() != -1
            loss = policy_gradient.flatten()[mask].sum()

            ## backprop
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            ## what are we recording against here?
            ## log for post training review
            self.log_batch_results(
                batch=batch,
                log_probs=log_probs.mean().cpu().detach().numpy(),
                rewards=rewards.flatten()[mask].sum().cpu().numpy() / (dones.flatten()[mask].sum().cpu().numpy()),
                returns=returns.flatten()[mask].sum().cpu().numpy()/  (dones.flatten()[mask].sum().cpu().numpy()),
                policy_gradient=loss.cpu().detach().numpy(),
                policy_entropy=(torch.exp(log_probs)*log_probs).mean().cpu().detach().numpy(),
                episode_len = (1-dones).flatten()[mask].sum().cpu().numpy() / dones.flatten()[mask].sum().cpu().numpy() 
            )

            ## run some evaluation (not strictly necessary given on-policy alg)
            ## done to be consistent with off policy and to separate
            ## training batch_size from num_eval_episodes
            if batch % eval_freq == 0:
                print(f'evaluating at {batch*steps_per_iter*num_envs}')
                self.eval_results[batch*steps_per_iter*num_envs] = self.evaluate(
                    env_name, self.num_eval_episodes
                )

            ## clear out records for next batch
            self.init_records()

    def act(self, x):
        action_dist = self.policy(torch.tensor(x).to(self.device))
        action = action_dist.sample()
        return action, action_dist.log_prob(action)