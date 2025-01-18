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


class ScaledDotProductAttention(nn.Module):
    """
    Implements scaled dot-product attention.
    """
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, n_heads, seq_len, d_k).
            key (torch.Tensor): Key tensor of shape (batch_size, n_heads, seq_len, d_k).
            value (torch.Tensor): Value tensor of shape (batch_size, n_heads, seq_len, d_v).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, seq_len, seq_len).

        Returns:
            torch.Tensor: Attention output.
            torch.Tensor: Attention weights.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot-product
        if mask is not None:
            scores = scores.masked_fill(0 == mask, -1e9)  # Apply mask

        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention mechanism.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Attention output.
        """
        batch_size = query.size(0)

        # Linear projections
        query = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.attention(query, key, value, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.w_o(attn_output)

        # Apply dropout and residual connection
        output = self.dropout(output)
        output = self.layer_norm(output + query.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model))

        return output
