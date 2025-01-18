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

from src.models.feed_forward import FeedForward
from src.models.multi_head_attention import MultiHeadAttention
from src.models.positional_encoding import PositionalEncoding
from src.models.transformer_decoder import TransformerDecoderBlock, TransformerDecoder
from src.models.transformer_encoder import TransformerEncoderBlock, TransformerEncoder
from src.models.transformer_model import TransformerModel

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


def test_transformer_encoder_block():
    d_model = 64
    n_heads = 8
    d_ff = 256
    seq_len = 10
    batch_size = 32

    x = torch.rand(batch_size, seq_len, d_model)
    encoder_block = TransformerEncoderBlock(d_model, n_heads, d_ff)
    output = encoder_block(x)

    assert output.shape == x.shape, "Output shape mismatch in Encoder Block"
    print("Transformer Encoder Block test passed!")


def test_transformer_encoder():
    d_model = 64
    n_heads = 8
    d_ff = 256
    num_layers = 3
    seq_len = 10
    batch_size = 32

    x = torch.rand(batch_size, seq_len, d_model)
    encoder = TransformerEncoder(d_model, n_heads, d_ff, num_layers)
    output = encoder(x)

    assert output.shape == x.shape, "Output shape mismatch in Encoder"
    print("Transformer Encoder test passed!")


def test_transformer_decoder_block():
    d_model = 64
    n_heads = 8
    d_ff = 256
    seq_len = 10
    batch_size = 32

    tgt = torch.rand(batch_size, seq_len, d_model)
    memory = torch.rand(batch_size, seq_len, d_model)
    decoder_block = TransformerDecoderBlock(d_model, n_heads, d_ff)
    output = decoder_block(tgt, memory)

    assert output.shape == tgt.shape, "Output shape mismatch in Decoder Block"
    print("Transformer Decoder Block test passed!")


def test_transformer_decoder():
    d_model = 64
    n_heads = 8
    d_ff = 256
    num_layers = 3
    seq_len = 10
    batch_size = 32

    tgt = torch.rand(batch_size, seq_len, d_model)
    memory = torch.rand(batch_size, seq_len, d_model)
    decoder = TransformerDecoder(d_model, n_heads, d_ff, num_layers)
    output = decoder(tgt, memory)

    assert output.shape == tgt.shape, "Output shape mismatch in Decoder"
    print("Transformer Decoder test passed!")

def test_transformer_model():
    input_dim = 4
    output_dim = 3
    d_model = 64
    n_heads = 8
    d_ff = 256
    num_encoder_layers = 3
    num_decoder_layers = 3
    src_len = 10
    tgt_len = 5
    batch_size = 32

    src = torch.rand(batch_size, src_len, input_dim)
    tgt = torch.rand(batch_size, tgt_len, output_dim)
    model = TransformerModel(input_dim, output_dim, d_model, n_heads, d_ff, num_encoder_layers, num_decoder_layers)
    output = model(src, tgt)

    assert output.shape == (batch_size, tgt_len, output_dim), "Output shape mismatch in Transformer Model"
    print("Transformer Model test passed!")

if __name__ == "__main__":
    test_positional_encoding()
    test_multi_head_attention()
    test_feed_forward()
    test_transformer_encoder_block()
    test_transformer_encoder()
    test_transformer_decoder_block()
    test_transformer_decoder()
    test_transformer_model()
