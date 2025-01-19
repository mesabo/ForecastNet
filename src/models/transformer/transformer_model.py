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

from src.models.transformer.positional_encoding import PositionalEncoding
from src.models.transformer.transformer_encoder import TransformerEncoder
from src.models.transformer.transformer_decoder import TransformerDecoder

class TransformerModel(nn.Module):
    """
    Full Transformer Model integrating Encoder and Decoder for sequence-to-sequence tasks.

    Args:
        input_dim (int): Dimensionality of input features.
        output_dim (int): Dimensionality of output features.
        d_model (int): Dimensionality of embedding space.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimensionality of the feed-forward network.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        max_len (int): Maximum sequence length for positional encoding.
        dropout (float): Dropout probability.
    """
    def __init__(self, input_dim, output_dim, d_model, n_heads, d_ff, num_encoder_layers, num_decoder_layers, max_len=5000, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.output_embedding = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, num_encoder_layers, max_len, dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, d_ff, num_decoder_layers, max_len, dropout)
        self.fc_out = nn.Linear(d_model, 1)  # Project to single value per timestep

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Forward pass for the Transformer Model.

        Args:
            src (torch.Tensor): Source input tensor (batch_size, src_len, input_dim).
            tgt (torch.Tensor): Target input tensor (batch_size, tgt_len).
            src_mask (torch.Tensor): Source attention mask.
            tgt_mask (torch.Tensor): Target attention mask.
            memory_mask (torch.Tensor): Memory attention mask.

        Returns:
            torch.Tensor: Predicted output (batch_size, tgt_len, output_dim).
        """
        src = self.input_embedding(src)
        src = self.positional_encoding(src)

        tgt = tgt.unsqueeze(-1)
        tgt = tgt.view(-1, tgt.size(-1))
        tgt = self.output_embedding(tgt)
        tgt = tgt.view(src.size(0), -1, self.d_model)
        tgt = self.positional_encoding(tgt)

        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        predictions = self.fc_out(output)  # Shape: (batch_size, tgt_len, 1)
        return predictions.squeeze(-1)  # Shape: (batch_size, tgt_len)