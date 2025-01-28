import itertools
import numpy as np

def get_subsets(L):
    """returns all possible subsets of L"""
    all_subsets = []
    for k in range(len(L)+1):
        subsets = itertools.combinations(L, k) #subsets of size k
        all_subsets.append(list(subsets))
    return all_subsets

def sample_subset(L):
    """returns a random subset of L of random size"""
    return list(np.sort(np.random.choice(L, np.random.randint(0,len(L)+1),replace=False)))

def get_remaining(L, player):
    """
    returns indices in L excluding player (list of indices)
    """
    return [el for el in L if el not in player]


def sample_coalitions(L, player, samples=0, aggregation=0, force=False):
    """
    return random coalitions in L excluding player
    player: list of indices = current player
    samples: number of random samples
        0 : no randomness
    aggregation:
        0 : all remaining subsets
        1 : remaining subset = other player (=> no randomness)
    
    """
    remaining = get_remaining(L, player)
    if aggregation == 1:
        return [[], remaining, L]
    elif samples == 0:
        subsets = get_subsets(remaining)
        return subsets
    else:
        sampled_coalitions = []
        if force:
            sampled_coalitions.append([]) #empty coalition
            sampled_coalitions.append(L.copy()) #full coalition
            samples = samples - 2
        for _ in range(samples):
            sampled_coalitions.append(sample_subset(remaining))
        return sampled_coalitions