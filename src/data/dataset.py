import torch
from torch.utils.data import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
  """dataset of multiple individuals"""
  def __init__(self, values, datetimes, context=None, lags=48, horizon=24,
               by_date=True, return_all_individuals=True, context_by_individuals=False):#, steps=None, seed=None, shuffle=False):    
    """
    values (N_individuals, dim_values, dates):  past target values 
    datetimes (dates): list of dates in datetime Y-m-d H:M:S format
    context (N_contexts, dim_context, dates): exogenous variates  e.g N_contexts=1 or N_contexts=N_individuals
    lags (int): size of lookback window
    horizon (int): size of target horizon
    by_date (bool): access items by date and random or all individuals
    return_all_individuals (bool): return all individuals or a random
    context_by_individuals(bool):  return one context per individual or all
    """
    super(TimeSeriesDataset, self).__init__()

    self.values, self.context = values, context
    self.lags, self.horizon = lags, horizon 
        
    self.individuals, self.dim_values, self.dates = self.values.shape
    self.contexts, self.dim_context, _dates = self.context.shape
    
    assert _dates == self.dates, "not the same dates in values and context"
    assert self.dates > self.lag + self.horizon, "not enough dates for this lag and horizon"
    
    self.datetimes = datetimes
    self.by_date = by_date
    self.return_all_individuals, self.context_by_individuals = return_all_individuals, context_by_individuals


  def shape(self):
    return (self.individuals, self.dim_values, self.dates), (self.contexts, self.dim_context, self.dates)

  def __len__(self):
    if self.by_date:
       return self.dates
    else:
       return self.N_individuals
    
  def __getitem__(self, idx):
    """    """

    if self.by_date:
        if self.return_all_individuals: #1 batch = all individuals, batch of dates
            values = self.values[:, :, idx + self.lags + self.horizon] # (individuals, dim_values, lags+horizon)
            context = self.context[:, :, idx + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)
            inputs = values[:, :, :self.lag] # (individuals, dim, lag)
            target = values[:, :, self.lag:] # (individuals, dim, horizon)
        else: #1 batch = 1 individual, batch of dates
            if self.seed is not None:
                np.random.seed(self.seed)
            indiv = np.random.randint(self.individuals)
            values = self.values[indiv, :, idx + self.lags + self.horizon] # (dim_values, lags+horizon)
            context = self.context[indiv, :, idx + self.lags + self.horizon] # (dim_context, lags+horizon)
            inputs = values[:, :, :self.lag] # (dim, lag)
            target = values[:, :, self.lag:] # (dim, horizon)

    else: #1 batch = batch of individuals, random date
        if self.seed is not None:
            np.random.seed(self.seed)
        t = np.random.randint(self.dates - self.lags - self.horizon)
        values = self.values[idx, :, t + self.lags + self.horizon] # (dim_values, lags+horizon)
        if self.context_by_individuals:
            context = self.context[idx, :, t + self.lags + self.horizon] # (dim_context, lags+horizon)
        else:
            context = self.context[:, :, t + self.lags + self.horizon] # (contexts, dim_context, lags+horizon)
        inputs = values[:, :, :self.lag] # (dim, lag)
        target = values[:, :, self.lag:] # (dim, horizon)

    return inputs, context, target

