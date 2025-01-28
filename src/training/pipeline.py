import torch
import torch.nn as nn
import torch.optim as optim

from time import perf_counter

from src.training.utils import normalize


def train_model(model, train_loader, valid_loader, device, lr=0.0001, do_print=True, normal=True, schedule=False, n_evals=100, n_prints=10, loss_name="mse"):
    """trains model and returns model, train and valid losses"""
    if loss_name is None or loss_name=="mse":
        criterion = nn.MSELoss()
    elif loss_name=="huber":
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unrecognized loss name")
    if normal:
        loss_name = "normalized_" + loss_name

    model.to(device)    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    steps = len(train_loader)
    eval_freq = max(steps // n_evals, 1)
    print_freq = max(steps // n_prints, 1)

    if schedule:
        lambda_lr = lambda epoch: 1 - (epoch / steps) #linear scheduler from 1 to 0
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    if do_print:
        print(f"Using device: {device}")
        print(f"Number of steps (batches) : {len(train_loader)}")

    train_losses = []
    valid_losses = []

    t1 = perf_counter()

    #training
    i = 0
    model.train()
    for X_batch, context_batch, y_batch in train_loader:
        X_batch, context_batch, y_batch = X_batch.to(device), context_batch.to(device), y_batch.to(device)          
        X_batch_normalized, mean, std = normalize(X_batch, return_stats=True) #mean (B,dim,1)
        
        optimizer.zero_grad()
        if normal: #normalized inputs to model
            predictions = model(X_batch_normalized, context_batch) # (B, dim, horizon)
        else:
            predictions = model(X_batch, context_batch)

        if normal: #output is in normalized space
            y_batch_normalized = (y_batch - mean) / std
            loss = criterion(predictions, y_batch_normalized) # mean over 1/B * 1/dim * 1/horizon
        else:
            loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item()) #loss of batch
        if schedule:
            scheduler.step()
        
        if i%(eval_freq)==0 or i==steps-1:
            valid_loss = eval_model(model, valid_loader, device, normal=normal) #loss over bs * valid_steps dates
            valid_losses.append(valid_loss)
            if i%(print_freq)==0 or i==steps-1:
                print(f"Step {i} - valid {loss_name} : {valid_loss:.4f}")
            model.train()
        i+=1

    t2 = perf_counter()
    model.eval()
    if do_print:
        print(f"Training done in {(t2-t1)/60:.3f} min")
    return model, train_losses, valid_losses


def eval_model(model, loader, device, normal=True, loss=None, return_all=False):
    """evaluates model on loader and returns mean loss
    if return_all, all individual losses
    """
    if loss is None:
        loss = nn.MSELoss(reduction="none")

    losses = []
    normalized_losses = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for X_batch, context_batch, y_batch in loader:
            X_batch, context_batch, y_batch = X_batch.to(device), context_batch.to(device), y_batch.to(device)            
            X_batch_normalized, mean, std = normalize(X_batch, return_stats=True)
            
            if normal:
                predictions_normalized = model(X_batch_normalized, context_batch)
                predictions = predictions_normalized*std + mean
            else:
                predictions = model(X_batch, context_batch) 
                predictions_normalized = (predictions - mean) / std
            y_batch_normalized = (y_batch - mean) / std
            
            losses.append(loss(predictions, y_batch))  #(B, dim, horizon)
            normalized_losses.append(loss(predictions_normalized, y_batch_normalized))

    losses = torch.cat(losses, dim=0).cpu() #(B * steps, dim, horizon)
    normalized_losses = torch.cat(normalized_losses, dim=0).cpu()

    if return_all:
        return losses, normalized_losses
    else:
        if normal:
            return normalized_losses.mean().item()
        else:
            return losses.mean().item()
