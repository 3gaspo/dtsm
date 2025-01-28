import torch.nn as nn
import torch.optim as optim
import torch
import os
import pandas as pd 
import json

def get_default_params(model, N_epochs, lr, get_scheduler=False):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  if get_scheduler:
    lambda_lr = lambda epoch: 1 - (epoch / N_epochs) #linear scheduler from 1 to 0
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    return {"criterion":criterion, "optimizer":optimizer, "scheduler":scheduler}
  else:
    return {"criterion":criterion, "optimizer":optimizer}

def save_results(metric, path, name, model_name, metric_name):
    """adds accuracy result to pandas file"""
    file_path = path + name
    if os.path.exists(file_path):
      with open(file_path, "r") as file:
        dico = json.load(file)
    else:
      dico = {}
    
    model_dico = dico.get(model_name, {})
    model_dico[metric_name] = metric
    dico[model_name] = model_dico
    with open(file_path, "w") as file:
      json.dump(dico, file, indent=4)



def normalize(X, return_stats=False):
  """
  X: tensor (B, dim, features)
  normalize for each B
  """
  mean = X.mean(dim=-1, keepdim=True)
  std =  X.std(dim=-1, keepdim=True)
  std = torch.where(std != 0, std, 1)
  
  X_normalized = (X - mean) / std

  if return_stats:
    return X_normalized, mean, std
  return X_normalized
