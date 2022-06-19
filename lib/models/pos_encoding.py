import torch
import numpy as np
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, enc_dim, in_dim=None, enc_type='fourier', max_freq=10, freq_scale=1.0, dropout=None, concat=True, learnable_pos_index=None):
        super(PositionalEncoding, self).__init__()
        self.enc_dim = enc_dim
        self.in_dim = enc_dim if in_dim is None else in_dim
        self.enc_type = enc_type
        self.max_freq = max_freq
        self.freq_scale = freq_scale
        self.concat = concat
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        if concat:
            self.fc = nn.Linear(self.enc_dim + self.in_dim, self.enc_dim)
        if learnable_pos_index is not None:
            if not isinstance(learnable_pos_index, torch.Tensor):
                learnable_pos_index = torch.LongTensor(learnable_pos_index)
            self.learnable_pos_index = learnable_pos_index
            self.learned_pe_res = nn.Parameter(torch.zeros(learnable_pos_index.shape[0], self.enc_dim))
        else:
            self.learnable_pos_index = None

    def original_positional_encoding(self, pos):
        pos = pos.unsqueeze(-1)
        mul_term = torch.exp(torch.arange(0, self.enc_dim, 2).to(pos.device) * (-np.log(10000.0) / self.enc_dim))
        pe = torch.stack([torch.sin(pos * mul_term), torch.cos(pos * mul_term)], dim=-1)
        pe = pe.view(-1, self.enc_dim)
        return pe

    def fourier_positional_encoding(self, pos):
        pos = pos.unsqueeze(-1)
        num_freq = self.enc_dim // 2
        mul_term = torch.exp(torch.arange(num_freq).to(pos.device) * (np.log(self.max_freq) / num_freq)) * self.freq_scale
        pe = torch.stack([torch.sin(pos * mul_term), torch.cos(pos * mul_term)], dim=-1)
        pe = pe.view(-1, self.enc_dim)
        return pe

    def generate_positional_encoding(self, pos):
        if self.enc_type == 'original':
            pe = self.original_positional_encoding(pos)
        elif self.enc_type == 'fourier':
            pe = self.fourier_positional_encoding(pos)
        else:
            raise ValueError('Unknown positional encoding type!')

        if self.learnable_pos_index is not None:
            pe[self.learnable_pos_index] += self.learned_pe_res
        return pe


    def forward(self, x=None, pos=None, seq_dim=0, x_shape=None, device=None, pos_offset=0):
        if x is not None:
            x_shape = x.shape 

        if pos is None:
            pos = torch.arange(x_shape[seq_dim], device=device if x is None else x.device)
            if pos_offset > 0:
                pos += pos_offset
        pe = self.generate_positional_encoding(pos)

        for _ in range(len(x_shape) - seq_dim - 2):
            pe = pe.unsqueeze(1)
        for _ in range(seq_dim):
            pe = pe.unsqueeze(0)
        
        if x is not None:
            if self.concat:
                pe_exp = pe.expand(x_shape[:-1] + (self.enc_dim,))
                x = torch.cat([x, pe_exp], dim=-1)
                x = self.fc(x)
            else:
                x = x + pe
        else:
            x = pe.expand(x_shape[:-1] + (self.enc_dim,))

        if self.dropout is not None:
            x = self.dropout(x)
        return x
