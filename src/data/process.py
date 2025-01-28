import torch
import numpy as np
from src.data.utils import get_temporal_features
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def set_random_data(path="datasets/", lag=168, horizon=24, do_plot=False):
    """gets a random individual and random window from dataset"""
    values = torch.load(path + "values.pt")
    datetimes = torch.load(path + "datetimes.pt")

    individuals, dim, dates = values.shape
    rand_indiv = np.random.randint(individuals)
    rand_date = np.random.randint(dates - (lag + horizon))

    inputs = values[rand_indiv, :, rand_date : rand_date+lag]
    target = values[rand_indiv, :, rand_date+lag : rand_date+lag+horizon]
    context = torch.tensor(get_temporal_features(datetimes[rand_date+lag]), dtype=torch.float32)

    torch.save(inputs, path + "rand_input.pt")
    torch.save(target, path + "rand_target.pt")
    torch.save(context, path + "rand_context.pt")


def fetch_example_data(path, names):
    """fetches example data"""
    if type(names) == list:
        dico = {}
        for name in names:
            input = torch.load(path + name + "_input.pt")
            context = torch.load(path + name + "_context.pt") 
            target = torch.load(path + name + "_target.pt")
            dico[name] = (input, context, target)
        return dico
    else:
        input = torch.load(path + names + "_input.pt")
        context = torch.load(path + names + "_context.pt") 
        target = torch.load(path + names + "_target.pt")
        return (input, context, target) 

def rename_example_data(path, name, new_dir):
    """renames example data to new name (to create new examples)"""
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    os.rename(path + "example.pdf", f"{new_dir}{name}_example.pdf")
    os.rename(path + "normal_example.pdf", f"{new_dir}{name}_normal_example.pdf")
    os.rename(path + "rand_input.pt", f"{new_dir}{name}_input.pt")
    os.rename(path + "rand_context.pt", f"{new_dir}{name}_context.pt")
    os.rename(path + "rand_target.pt", f"{new_dir}{name}_target.pt")


def get_stats(values, stat, dim=0):
    """returns tensor of given stats for a loader
    values (Nindiv, Ndim, Ndates)
    """
    if stat == "mean":
        values_stat = values.mean(axis=-1) #(Nindiv, Ndim)
        total_stat = values.mean()
    elif stat == "max":
        values_stat, _ = values.max(axis=-1) #(Nindiv, Ndim)
        total_stat = values.max()
    else:
        raise ValueError("Unrecognized stat name")
    return values_stat[:, dim], total_stat #(Nindiv)