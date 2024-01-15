import torch
from torch import nn
from torch.optim import Adam


class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=-1)
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.layers[-2].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AdversarialDiscriminator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, gamma: float, lr: float
                 ) -> None:
        assert 0. < gamma < 1.
        assert 0. < lr < 1.
        super().__init__()
        self.g = nn.Sequential(
            nn.Linear(input_dim + output_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.h = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.gamma = gamma
        self.output_dim = output_dim
        self.optimizer = Adam(self.parameters(), lr)
        for layer in self.g:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.h:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def f(self, state: torch.Tensor, action: torch.Tensor,
          next_state: torch.Tensor) -> torch.Tensor:
        return self.g(torch.concat((
            state, nn.functional.one_hot(action, self.output_dim).squeeze(1)),
            dim=1)) + self.gamma * self.h(next_state) - self.h(state)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                next_state: torch.Tensor, action_prob: torch.Tensor
                ) -> torch.Tensor:
        e_f = torch.exp(self.f(state, action, next_state))
        return e_f / (e_f + action_prob)

    def get_reward(self, state: torch.Tensor, action: torch.Tensor,
                   next_state: torch.Tensor, action_prob: torch.Tensor
                   ) -> torch.Tensor:
        d = self.forward(state, action, next_state, action_prob)
        return (torch.log(d) - torch.log(1. - d)).detach()
