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

from src.models.transformer.feed_forward import FeedForward
from src.models.transformer.multi_head_attention import MultiHeadAttention
from src.models.transformer.positional_encoding import PositionalEncoding


class TransformerDecoderBlock(nn.Module):
    """
    A single Transformer Decoder Block.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the Transformer Decoder Block.

        Args:
            tgt (torch.Tensor): Target sequence tensor (batch_size, seq_len, d_model).
            memory (torch.Tensor): Encoder output tensor (batch_size, seq_len, d_model).
            tgt_mask (torch.Tensor): Target attention mask.
            memory_mask (torch.Tensor): Memory attention mask.

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, d_model).
        """
        # Masked Multi-Head Attention
        residual = tgt
        x = self.masked_attn(tgt, tgt, tgt, tgt_mask)
        x = self.dropout(x)
        x = self.layer_norm1(x + residual)

        # Cross-Attention
        residual = x
        x = self.cross_attn(x, memory, memory, memory_mask)
        x = self.dropout(x)
        x = self.layer_norm2(x + residual)

        # Feed-Forward Network
        residual = x
        x = self.feed_forward(x)
        x = self.layer_norm3(x + residual)

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder consisting of multiple Decoder Blocks.

    Args:
        d_model (int): Dimensionality of the input embeddings.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network.
        num_layers (int): Number of decoder blocks.
        max_len (int): Maximum sequence length.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model, n_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the Transformer Decoder.

        Args:
            tgt (torch.Tensor): Target sequence tensor (batch_size, seq_len, d_model).
            memory (torch.Tensor): Encoder output tensor (batch_size, seq_len, d_model).
            tgt_mask (torch.Tensor): Target attention mask.
            memory_mask (torch.Tensor): Memory attention mask.

        Returns:
            torch.Tensor: Decoded output (batch_size, seq_len, d_model).
        """
        tgt = self.positional_encoding(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)

        return self.layer_norm(tgt)
