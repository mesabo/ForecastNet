#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/15/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for adding positional information to embeddings.

    Args:
        d_model (int): Dimensionality of the embedding space.
        max_len (int): Maximum sequence length for the positional encoding.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encodings
        self.register_buffer("pe", self._generate_pe(d_model, max_len))

    def _generate_pe(self, d_model, max_len):
        """
        Generate sinusoidal positional encodings.

        Args:
            d_model (int): Dimensionality of the embedding space.
            max_len (int): Maximum sequence length.

        Returns:
            torch.Tensor: Positional encodings of shape (1, max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Sinusoidal on even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine on odd indices
        return pe.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        """
        Add positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Input tensor with positional encodings added.
        """
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Input sequence length ({x.size(1)}) exceeds the maximum allowed length ({self.pe.size(1)}).")

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)