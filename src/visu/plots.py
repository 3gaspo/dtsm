import matplotlib.pyplot as plt
import numpy as np
from src.data.process import get_stats

def plot_losses(train_losses, valid_losses=None, path="", name="losses.pdf", title="Losses", logscale=True, n_evals=10):
    """plots losses during training"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    if valid_losses is not None:
        plt.plot(train_losses, label="train")
        test_freq = len(train_losses) / n_evals
        T = [test_freq * k for k in range(n_evals)] + [len(train_losses)-1]
        plt.plot(T, valid_losses, label="valid")
    else:
        plt.plot(train_losses, label="valid")
    if logscale:
      plt.yscale('log')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.savefig(path + name)


def plot_multi_losses(losses_dict, path="", name="losses.pdf", title="Losses", logscale=True, n_evals=10):
    """plots losses during training"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for expe_name, (train_losses, valid_losses) in losses_dict.items():
        if valid_losses is not None:
            plt.plot(train_losses, label=f"{expe_name} train")
            test_freq = len(train_losses) / n_evals
            T = [test_freq * k for k in range(n_evals)] + [len(train_losses)-1]
            plt.plot(T, valid_losses, label=f"{expe_name} valid")
        else:
            plt.plot(train_losses, label=f"{expe_name} valid")
        if logscale:
            plt.yscale('log')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    fig.tight_layout()
    plt.savefig(path + name)

def plot_pred(x, y, pred, path="", name="prediction.pdf", title="Predictions"):
    """plots example prediction"""
    plt.clf()
    lag = len(x)
    horizon = len(y)
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(lag), x, label="Lookback")
    plt.plot(range(lag, lag+horizon), pred, label="Prediction")
    plt.plot(range(lag, lag+horizon), y, label="Horizon")
    plt.axvline(x=lag, color='black', linestyle='--')
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path + name)

def plot_example(x, y, path="", name="example.pdf", title="Example"):
    """plots example data"""
    plt.clf()
    lag = len(x)
    horizon = len(y)
    fig = plt.figure(figsize=(20,5))
    plt.plot(range(lag), x, label="Lookback")
    plt.plot(range(lag, lag+horizon), y, label="Horizon")
    plt.axvline(x=lag, color='black', linestyle='--')
    plt.legend(bbox_to_anchor=(0.5, -0.15), ncol=3, loc='center', fontsize=14)
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path + name)

def plot_errors(losses, path="", name="errors.pdf", title="Loss distribution"):
    """plots histogram of errors"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    plt.hist(losses, bins=100)
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Losses")
    plt.ylabel("Frequency")
    plt.savefig(path + name)


def plot_horizon_errors(losses, path="", name="horizon.pdf", title="Mean errors by horizon"):
    """plots errors according to horizon"""
    plt.clf()
    fig = plt.figure(figsize=(15,5))
    plt.bar(range(len(losses)), losses)
    plt.title(title)
    plt.xlabel("Horizon")
    plt.ylabel("Mean error")
    plt.savefig(path + name)

def plot_stats(splits_dict, stat, path="", name="stats.pdf", dim=0):
    """plots stats of datasets"""
    plt.clf()
    fig = plt.figure(figsize=(10,5))
    for split_name, split_dict in splits_dict.items():
        stat_values, total_stat = get_stats(split_dict["values"], stat, dim)
        bins = np.logspace(-2, 6, 100)
        plt.hist(stat_values, label= f"{split_name} - {stat}={total_stat:.2f}", bins=bins, density=True, alpha=0.5)
    plt.legend()
    plt.title(f"{stat} distribution")
    plt.xlabel(f"{stat}")
    plt.xscale("log")
    plt.ylabel("Counts")
    plt.savefig(path + name)