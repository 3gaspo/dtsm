import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from src.data.utils import get_temporal_features, train_test_split

class IndividualDataset(Dataset):
  """dataset for a given individual"""
  def __init__(self, values, datetimes, lag, horizon, indiv=None, steps=None, seed=None, shuffle=False):    
    """
    values : past values (individuals, dim, dates)
    datetimes : (dates)
    indiv : index of individual
    """
    super(IndividualDataset, self).__init__()
    self.lag, self.horizon = lag, horizon #size of look-back window, size of horizon to predict
    
    if indiv is not None:
      assert(type(indiv) == int)
      self.values = values[indiv] # (dim, dates)
    else:
      self.values = values.sum(dim=1)
    self.steps, self.seed, self.shuffle = steps, seed, shuffle

    self.dim, self.dates = self.values.shape
    self.datetimes = datetimes
    assert self.dates > self.lag + self.horizon, "not enough dates for this lag and horizon"
    self.window = self.dates - (self.lag + self.horizon)

  def shape(self):
    return self.dim, self.dates

  def __len__(self):
    if self.steps is None:
      return self.window
    else:
      return self.steps

  def __getitem__(self, idx):
    """
    Returns past values, context of idx and horizon
    """
    if self.shuffle:
      if self.seed is not None:
        np.random.seed(self.seed)
      t = np.random.randint(self.lag, self.dates - self.horizon) #random start point
    else:
      t = idx % self.window
    values = self.values[:, t-self.lag:t+self.horizon] # (dim, lag + horizon)  #lookback window + horizon

    inputs = values[:, :self.lag,] # (dim, lag) #lookback window
    target = values[:, self.lag:] # (dim, horizon) #horizon
    context = torch.tensor(get_temporal_features(self.datetimes[t]), dtype=torch.float32) # (context)

    return inputs, context, target


class SeriesDataset(Dataset):
  """dataset of multiple individuals"""
  def __init__(self, values, datetimes, lag, horizon, steps=None, seed=None, shuffle=False):    
    """
    values: past values (individuals, dim, dates)
    datetimes : (dates)
    """
    super(SeriesDataset, self).__init__()
    self.lag, self.horizon = lag, horizon #size of look-back window, size of horizon to predict
    self.values = values
    self.individuals, self.dim, self.dates = self.values.shape
    self.datetimes = datetimes
    assert self.dates > self.lag + self.horizon, "not enough dates for this lag and horizon"
    self.steps, self.seed, self.shuffle = steps, seed, shuffle


  def shape(self):
    return self.individuals, self.dim, self.dates

  def __len__(self):
    if self.steps is None:
      return self.individuals
    else:
      return self.steps
    
  def __getitem__(self, idx):
    """
    Returns random data for random date of individual
    """
    if self.seed is not None:
      np.random.seed(self.seed)
    t = np.random.randint(self.lag, self.dates - self.horizon) #random start point
    if self.shuffle:
      indiv = np.random.randint(0, self.individuals)
    else:
      indiv = idx % self.individuals
    values = self.values[indiv, :, t-self.lag:t+self.horizon] # (dim, lag + horizon)  #lookback window + horizon

    inputs = values[:, :self.lag,] # (dim, lag) #lookback window
    target = values[:, self.lag:] # (dim, horizon) #horizon
    context = torch.tensor(get_temporal_features(self.datetimes[t]), dtype=torch.float32) # (context)

    return inputs, context, target
  

def get_data_splits(values, datetimes, indiv_split=None, date_split=None):
  """returns dictionnary of 4 splits (indiv and time)"""
  splits = train_test_split(values, datetimes, indiv_split, date_split, seed=None)
  datasplits = {}
  for split_key, (split_values, split_dates) in splits.items():
    datasplits[split_key] = {"values":split_values, "dates":split_dates}
  return datasplits

def get_data_loaders(values, datetimes, steps, indiv_split=None, date_split=None, indiv=None, lag=24*7, horizon=24, batch_size=32, seed=None, valid_steps=100, test_steps=100):
  """returns split dataloaders"""
  splits = train_test_split(values, datetimes, indiv_split, date_split, seed=None)
  dataloader = {}
  train_size = steps * batch_size
  valid_size = valid_steps * batch_size
  test_size = test_steps * batch_size
  dataset_sizes = {"train":train_size, "valid":valid_size, "split_3": valid_size, "test":test_size}
  for split_key, (split_values, split_dates) in splits.items():
    dataset_size = dataset_sizes[split_key]
    if indiv is not None:
      split_dataset = IndividualDataset(split_values, split_dates, lag, horizon, indiv, dataset_size, seed)
    else:
      split_dataset = SeriesDataset(split_values, split_dates, lag, horizon, dataset_size, seed)
    dataloader[split_key] = DataLoader(split_dataset, batch_size=batch_size, shuffle=True)
  return dataloader
