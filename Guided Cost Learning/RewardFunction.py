import math
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
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                deno = math.sqrt(128)
                torch.nn.init.uniform_(layer.weight, -1 / deno, 1 / deno)
                layer.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
