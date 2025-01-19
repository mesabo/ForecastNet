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
from src.models.multi_head_attention import MultiHeadAttention
from src.models.feed_forward import FeedForward
from src.models.positional_encoding import PositionalEncoding


class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer Encoder Block combining Multi-Head Attention,
    Feed-Forward, and Residual Connections.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the Transformer Encoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-Attention and Residual Connection
        residual = x
        x = self.attention(x, x, x, mask)
        x = self.dropout(x)
        x = self.layer_norm1(x + residual)

        # Feed-Forward and Residual Connection
        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm2(x + residual)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple Encoder Blocks.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network.
        num_layers (int): Number of encoder blocks.
        max_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model, n_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Forward pass for the Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): Attention mask of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Encoded output of shape (batch_size, seq_len, d_model).
        """
        x = self.positional_encoding(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        return self.layer_norm(x)
