import torch
from torch import nn


class RewardFunction(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(RewardFunction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
