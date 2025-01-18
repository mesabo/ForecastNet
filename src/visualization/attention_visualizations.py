#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/19/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_attention_weights(attention_weights, input_tokens, output_tokens, output_path):
    """
    Visualizes attention weights as a heatmap.

    Args:
        attention_weights (torch.Tensor): Attention weights with shape (n_heads, tgt_len, src_len).
        input_tokens (list): List of input token names (e.g., timestamps or features).
        output_tokens (list): List of output token names (e.g., forecast steps).
        output_path (str or Path): Path to save the attention visualization plot.
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    n_heads, tgt_len, src_len = attention_weights.shape

    for head in range(n_heads):
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_weights[head],
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap="coolwarm",
            cbar=True,
            annot=False,
        )
        plt.title(f"Attention Weights - Head {head + 1}")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")
        plt.tight_layout()
        plt.savefig(output_path / f"attention_head_{head + 1}.png")
        plt.show()


def aggregate_attention_weights(attention_weights):
    """
    Aggregates attention weights across heads by averaging.

    Args:
        attention_weights (torch.Tensor): Attention weights with shape (n_heads, tgt_len, src_len).

    Returns:
        np.ndarray: Aggregated attention weights with shape (tgt_len, src_len).
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    return np.mean(attention_weights, axis=0)


def plot_aggregated_attention_weights(attention_weights, input_tokens, output_tokens, output_path):
    """
    Visualizes aggregated attention weights across all heads.

    Args:
        attention_weights (torch.Tensor): Attention weights with shape (n_heads, tgt_len, src_len).
        input_tokens (list): List of input token names.
        output_tokens (list): List of output token names.
        output_path (str or Path): Path to save the aggregated attention visualization.
    """
    aggregated_weights = aggregate_attention_weights(attention_weights)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        aggregated_weights,
        xticklabels=input_tokens,
        yticklabels=output_tokens,
        cmap="coolwarm",
        cbar=True,
        annot=False,
    )
    plt.title("Aggregated Attention Weights")
    plt.xlabel("Input Tokens")
    plt.ylabel("Output Tokens")
    plt.tight_layout()
    plt.savefig(output_path / "aggregated_attention.png")
    plt.show()