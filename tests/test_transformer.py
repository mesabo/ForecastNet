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
from src.models.positional_encoding import PositionalEncoding
from src.models.multi_head_attention import MultiHeadAttention
from src.models.feed_forward import FeedForward


def test_positional_encoding():
    d_model = 64
    seq_len = 50
    batch_size = 32

    # Create dummy input
    x = torch.zeros(batch_size, seq_len, d_model)

    # Initialize positional encoding
    pe = PositionalEncoding(d_model)

    # Forward pass
    output = pe(x)

    assert output.shape == x.shape, "Output shape mismatch"
    print("Positional Encoding test passed!")


def test_multi_head_attention():
    d_model = 64
    n_heads = 8
    seq_len = 10
    batch_size = 32

    # Dummy input tensors
    query = torch.rand(batch_size, seq_len, d_model)
    key = torch.rand(batch_size, seq_len, d_model)
    value = torch.rand(batch_size, seq_len, d_model)

    # Initialize MultiHeadAttention
    mha = MultiHeadAttention(d_model, n_heads)

    # Forward pass
    output = mha(query, key, value)

    assert output.shape == query.shape, "Output shape mismatch"
    print("Multi-Head Attention test passed!")


def test_feed_forward():
    d_model = 64
    d_ff = 256
    seq_len = 10
    batch_size = 32

    # Dummy input tensor
    x = torch.rand(batch_size, seq_len, d_model)

    # Initialize FeedForward
    ff = FeedForward(d_model, d_ff)

    # Forward pass
    output = ff(x)

    assert output.shape == x.shape, "Output shape mismatch"
    print("Feed-Forward Layer test passed!")

# if __name__ == "__main__":
    # test_positional_encoding()
    # test_multi_head_attention()
    # test_feed_forward()
