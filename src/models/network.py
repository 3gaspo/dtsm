import torch
import torch.nn as nn
from src.training.utils import normalize

class MLP(nn.Module):
    def __init__(self, lag, hidden, horizon, dim=1, context=6, revin=0):
        super(MLP, self).__init__()
        self.lag, self.horizon  = lag, horizon
        self.hidden = hidden
        self.dim, self.context = dim, context
        self.revin = revin
        self.fc = nn.Sequential(
            nn.Linear(lag * dim + context, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * dim)
        )
        if revin:
            self.gamma = nn.Parameter(torch.ones(self.dim)) #(dim)
            self.beta = nn.Parameter(torch.zeros(self.dim)) #(dim)

    def forward(self, x, context=None):
        """
        x : past values (B, dim, lag)
        context : current context (B, context)
        """
        batch_size = x.shape[0]
        if self.revin == 1:
            gamma, beta = self.gamma.unsqueeze(0).unsqueeze(-1), self.beta.unsqueeze(0).unsqueeze(-1)
            x, mean, std = normalize(x, return_stats=True)
            x = gamma * x + beta
        input = x.view(batch_size, self.lag * self.dim) # (B, lag*dim)
        if  context is not None:
            input = torch.cat((input, context), dim=1) # (B, lag*dim+context)
        output = self.fc(input) # (B, horizon*dim)
        output = output.view(batch_size, self.dim, self.horizon) # (B, dim, horizon)
        if self.revin == 1:
            output = std * ((output - beta) / gamma) + mean
        return output