import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Actor (Policy) Model.
    Implements fully connected layers
    """

    def __init__(self, state_size, action_size, seed, hidden_layers = [64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int) : number of units of hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        layers = [state_size] + hidden_layers + [action_size]
        layers = zip(layers, layers[1:])
        self.hl = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in layers])
        self.name = "_".join([str(x) for x in hidden_layers])

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for linear in self.hl[:-1]:
            x = F.relu(linear(x))
        
        linear = self.hl[-1]

        return linear(x)
