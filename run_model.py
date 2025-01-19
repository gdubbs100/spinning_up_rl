import os
import argparse
import json

import torch
import torch.optim as optim
import gymnasium as gym
import pandas as pd

from datetime import datetime

from algorithms.reinforce import ReinforceAgent
from networks.policy import DiscretePolicy, ContinuousPolicy

from algorithms.dqn import DQNAgent
from networks.value_networks import Qnet

from utils.training_utils import set_seed


## add argparse args to determine:
## - model, environment, n timesteps, hyper params
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='reinforce', help='specify algorithm')
parser.add_argument('--env_name', type=str, default='CartPole-v1', help='specify gymnasium env')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--log_dir', type=str, default='./logs', help='location to save runs - will be saved in a subfolder of this dir')

## training params
parser.add_argument('--num_envs', type=int, default=4, help='specify the number of environments to run in parallel')
parser.add_argument('--num_iters',type=int, default = 10, help='specify the number of update iterations - valid for any algorithm')
parser.add_argument('--steps_per_iter', type=int, default=500, help="length of rollout / number of env steps per iteration - valid for any algorithm")
parser.add_argument('--eval_freq', type=int, default=25, help='frequency of running evaluations -- valid for any algorithm')
parser.add_argument('--discount_rate', type=float, default=0.99, help='discount_rate for return calculation')
parser.add_argument('--learning_rate', type=float, default=0.01, help = 'learning rate for neural networks - currently valid for any algorithm')

## REINFORCE args
## none currently

## DQN args
parser.add_argument('--buffer_size', type=int, default=10000, help = 'number of observations stored in replay buffer')
parser.add_argument('--mini_batch_size', type=int, default=128, help='number of samples drawn from replay buffer per update')
parser.add_argument('--epsilon', type=float, default=0.9, help='epsilon value for epsilon-greedy exploration')

args = parser.parse_args()
if __name__ == "__main__":

    # print(f"args: {args}")
    timestamp = datetime.now().strftime(format='%Y%m%d%H%M%S')
    set_seed(args.seed)
    ## NOTE: this is not the actual env used for training
    env = gym.make(args.env_name)

    ## create agents
    if args.algorithm == 'reinforce':
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):
            ## TODO: create network parameters as an argparse input
            policy = DiscretePolicy(
                env.observation_space.shape[0],
                env.action_space.n,
                2*(env.action_space.n+env.observation_space.shape[0])
            )
        else:
            policy = ContinuousPolicy(
                env.observation_space.shape[0], 
                env.action_space.shape[0], 
                2*(env.action_space.shape[0]+env.observation_space.shape[0])
            )
        ## TODO: set optimiser as optional input arg
        agent = ReinforceAgent(
            policy, 
            policy_lr=args.learning_rate, 
            optimiser=optim.SGD,
            seed=args.seed
        )


        
    elif args.algorithm == 'dqn':
        print('running dqn')
        if isinstance(env.action_space, gym.spaces.discrete.Discrete):

            ## TODO: set network specs as input args
            q_net = Qnet(
                state_dim = env.observation_space.shape[0],
                action_dim = env.action_space.n,
                intermediate_dim=128,
                n_layers=2
            )

        else:
            raise ValueError("DQN does not support continuous action spaces...")
        
        ## TODO: set optimiser as optional input arg
        agent = DQNAgent(
            q_net, 
            lr=args.learning_rate, 
            optimiser=optim.AdamW, 
            buffer_size=args.buffer_size,
            mini_batch_size=args.mini_batch_size,
            epsilon=args.epsilon,
            discount_rate=args.discount_rate,
            seed=args.seed
        )
    else:
        raise ValueError(f"no algorithm named: {args.algorithm}...")

    ## train agents
    print(f"{timestamp}: training agent using {args.algorithm} on environment {args.env_name}...")
    agent.train(
        env_name=args.env_name, 
        num_envs=args.num_envs, 
        num_iters=args.num_iters, 
        steps_per_iter=args.steps_per_iter, 
        eval_freq=args.eval_freq
    )

    ## setup logging dir
    log_dir = f"{args.log_dir}/{args.env_name}/{args.algorithm}_{args.seed}/{timestamp}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    train_results = pd.DataFrame(agent.batch_results).T
    test_results = pd.DataFrame(
        agent.eval_results.values(), 
        index=agent.eval_results.keys(), 
        columns=['reward']
    )

    train_results.to_csv(log_dir + 'train_results.csv')
    test_results.to_csv(log_dir + 'test_results.csv')
    with open(log_dir + 'hyperparams.json', 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)

    print(agent.eval_results)

