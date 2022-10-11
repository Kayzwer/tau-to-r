import torch
from torch import nn


class RewardFunction(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(RewardFunction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)
                layer.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -self.layers(x)
