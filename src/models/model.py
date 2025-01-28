import torch
import torch.nn as nn

class TimeSeriesModel(nn.Module):
    def __init__(self, lags, horizon,
                 individuals_in, individuals_out, dim_values, context_in, dim_context):
        super(TimeSeriesModel, self).__init__()
        self.lags, self.horizon = lags, horizon
        self.individuals_in, self.individuals_out, self.context_in = individuals_in, individuals_out, context_in
        self.dim_values, self.dim_context = dim_values, dim_context

    def forward(self, x, context):
        """
        x (batch_size, individuals_in, dim_values, lags)
        context (batch_size, context_in, dim_context, lags+horizon)
        output (individuals_out, dim_values, horizon)
        """
        pass