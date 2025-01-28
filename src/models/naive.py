import torch.nn as nn

class persistence(nn.Module):
    """repeats last value"""
    def __init__(self, horizon):
        super(persistence, self).__init__()
        self.horizon = horizon

    def forward(self, x, context=None):
        """
        x : past values (B, dim, lag)
        context : current context (B, context)
        """
        past_values = x[:, :, -1].unsqueeze(2) # (B, dim, 1)
        output = past_values.repeat(1, 1, self.horizon) # (B, dim, horizon)
        return output
    
class repeat(nn.Module):
    """returns last values as horizon"""
    def __init__(self, horizon):
        super(repeat, self).__init__()
        self.horizon = horizon

    def forward(self, x, context=None):
        """
        x : past values (B, dim, lag)
        context : current context (B, context)
        """
        output = x[:, :, -self.horizon:] # (B, dim, horizon)
        return output


class lookback(nn.Module):
    """returns values a week ago (starts at idx)"""
    def __init__(self, idx, horizon):
        super(lookback, self).__init__()
        self.idx  = idx
        self.horizon = horizon

    def forward(self, x, context=None):
        """
        x : past values (B, dim, lag)
        context : current context (B, context)
        """
        output = x[:, :, self.idx:self.idx+self.horizon] # (B, dim, horizon)
        return output
    
class linear(nn.Module):
    """linear layer on lags"""
    def __init__(self, lag, horizon, dim):
        super(linear, self).__init__()
        self.lag  = lag
        self.dim = dim
        self.horizon = horizon

        self.fc = nn.Linear(lag * dim, horizon * dim)

    def forward(self, x, context=None):
        """
        x : past values (B, dim, lag)
        context : current context (B, context)
        """
        batch_size = x.shape[0]
        input = x.view(batch_size, self.lag * self.dim) # (B, lag*dim)
        output = self.fc(input) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        return output