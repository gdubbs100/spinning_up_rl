import torch
import random
import numpy as np

def set_seed(seed: int):
    print('Seeding random, torch, numpy...')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)


def calc_returns(
    rewards: torch.FloatTensor, 
    dones: torch.IntTensor, 
    discount_rate: float) -> torch.FloatTensor:
    """
    Generic calculation of returns
    """
    returns = torch.zeros_like(rewards)
    T = returns.size(-1)
    returns[...,-1] = rewards[...,-1] + discount_rate*returns[...,-1] * (1-dones[...,-1])
    for t in reversed(range(T-1)):
        returns[...,t] = rewards[...,t] + discount_rate*returns[...,t+1] * (1-dones[...,t])
    return returns