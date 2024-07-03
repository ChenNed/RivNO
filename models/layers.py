import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, InstanceNorm, BatchNorm
from torch_geometric.utils import dropout_edge, scatter
import torch.nn.functional as F


################################################################
# Activation functions
################################################################
def _get_act(act):
    if act == 'tanh':
        func = nn.Tanh
    elif act == 'gelu':
        func = nn.GELU
    elif act == 'relu':
        func = nn.ReLU
    elif act == 'elu':
        func = nn.ELU
    elif act == 'leaky_relu':
        func = nn.LeakyReLU
    elif act == 'swish':
        func = Swish
    elif act == 'sin':
        func = Sine
    else:
        raise ValueError(f'{act} is not supported')
    return func


class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(30 * x)


class Swish(nn.Module):
    """
    Swish activation function
    """

    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


################################################################
# MLP layers
################################################################

def linear_block(in_channel, out_channel, nonlinearity):
    block = nn.Sequential(
        nn.Linear(in_channel, out_channel),
        nonlinearity()
    )
    return block


class MLP(nn.Module):
    '''
    Fully connected layers with Tanh as nonlinearity
    Reproduced from PINNs Burger equation
    '''

    def __init__(self, nonlinearity, layers=[2, 10, 1]):
        super(MLP, self).__init__()

        if isinstance(nonlinearity, str):
            if nonlinearity == 'tanh':
                nonlinearity = nn.Tanh
            elif nonlinearity == 'relu':
                nonlinearity = nn.ReLU
            elif nonlinearity == 'sin':
                nonlinearity = Sine
            elif nonlinearity == 'swish':
                nonlinearity = Swish
            else:
                raise ValueError(f'{nonlinearity} is not supported')

        fc_list = [linear_block(in_size, out_size, nonlinearity)
                   for in_size, out_size in zip(layers, layers[1:-1])]
        fc_list.append(nn.Linear(layers[-2], layers[-1]))
        self.fc = nn.Sequential(*fc_list)

    def forward(self, x):
        return self.fc(x)


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1
        if isinstance(nonlinearity, str):
            if nonlinearity == 'tanh':
                nonlinearity = nn.Tanh
            elif nonlinearity == 'relu':
                nonlinearity = nn.ReLU
            elif nonlinearity == 'sin':
                nonlinearity = Sine
            elif nonlinearity == 'swish':
                nonlinearity = Swish
            else:
                raise ValueError(f'{nonlinearity} is not supported')

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


################################################################
# Message Passing layers
################################################################

class MPLayerv1(MessagePassing):
    """
    Message passing layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 n_variables: int,
                 act: str
                 ):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(MPLayerv1, self).__init__(aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.n_variables = n_variables

        self.act = _get_act(act)

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + 2 + 2 + n_variables, hidden_features),
                                           self.act()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           self.act()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          self.act()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          self.act()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)

        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class MPLayer(MessagePassing):
    """
    Message passing layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 n_variables: int,
                 act: str
                 ):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(MPLayer, self).__init__(aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.n_variables = n_variables

        self.act = _get_act(act)

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + 2 + n_variables, hidden_features),
                                           self.act()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           self.act()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          self.act()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          self.act()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)

        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class GNBlock(nn.Module):
    r"""A graph neural network block based on Battaglia et al. (2018) (https://arxiv.org/abs/1806.01261).

    Args:
        edge_mlp_args (Tuple): Arguments for the MLP used for updating the edge features.
        node_mlp_args (Tuple): Arguments for the MLP used for updating the node features.
        aggr (str, optional): The aggregation operator to use. Can be 'mean' or 'sum'. Defaults to 'mean'.

    Methods:
        reset_parameters(): Reinitializes the parameters of the MLPs.
        forward(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor: Computes the forward pass of the GN block.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 act: str,
                 n_variables: int,
                 aggr: str = 'mean'):
        super().__init__()
        self.act = _get_act(act)
        self.edge_mlp = nn.Sequential(nn.Linear(2 * in_features + 2, hidden_features),
                                      self.act(),
                                      nn.Linear(hidden_features, hidden_features),
                                      self.act(),
                                      nn.LayerNorm(hidden_features)
                                      )

        self.node_mlp = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                      self.act(),
                                      nn.Linear(hidden_features, out_features),
                                      self.act(),
                                      nn.LayerNorm(hidden_features)
                                      )

        # self.norm = InstanceNorm(hidden_features)
        self.aggr = aggr

    def reset_parameters(self):
        models = [model for model in [self.node_mlp, self.edge_mlp] if model is not None]
        for model in models:
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()

    def forward(self,
                v: torch.Tensor,
                e: torch.Tensor,
                variables: torch.Tensor,
                edge_index: torch.Tensor
                ) -> torch.Tensor:
        row, col = edge_index
        # Edge update
        e = self.edge_mlp(torch.cat((e, v[row], v[col]), dim=-1))
        # Edge aggregation
        aggr = scatter(e, col, dim=0, dim_size=v.size(0), reduce=self.aggr)
        # Node update
        v = self.node_mlp(torch.cat((aggr, v, variables), dim=-1))
        return v, e
