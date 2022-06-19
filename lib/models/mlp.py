# This script is borrowed and extended from https://github.com/Khrylx/AgentFormer/blob/main/model/common/mlp.py
# Adhere to their licence to use this script  


import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh', norm_type=None, num_norm_groups=16):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.out_dim = hidden_dims[-1]
        self.norm_type = norm_type
        self.affine_layers = nn.ModuleList()
        if norm_type is not None:
            self.norm_layers = nn.ModuleList()

        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            if norm_type == 'group_norm':
                self.norm_layers.append(nn.GroupNorm(num_norm_groups, nh))
            last_dim = nh

    def forward(self, x):
        for i, affine in enumerate(self.affine_layers):
            x = affine(x)
            if self.norm_type is not None:
                if len(x.shape) == 3 and self.norm_type == 'group_norm':
                    x = self.norm_layers[i](x.transpose(-1, -2)).transpose(-1, -2)
                else:
                    x = self.norm_layers[i](x)
            x = self.activation(x)
        return x

