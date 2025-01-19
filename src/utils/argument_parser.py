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

import argparse

def get_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Time-series forecasting with Transformers.")

    # Task choices
    parser.add_argument("--task", type=str, choices=["time_series", "classification", "regression"],
                        default="time_series", help="Task type (e.g., time_series, classification, regression).")

    # Test case choices
    parser.add_argument("--test_case", type=str,
                        choices=["generate_data", "preprocess_data", "train_transformer", "evaluate_transformer"],
                        default="generate_data", help="Test case to run.")

    # Dataset and paths
    parser.add_argument("--data_path", type=str, default="data/raw/time_series_data.csv", help="Path to dataset.")
    parser.add_argument("--save_path", type=str, help="Path to save models and outputs.")

    # Model and hyperparameters
    parser.add_argument("--model", type=str, choices=["transformer"], default="transformer", help="Model type.")
    parser.add_argument("--batch_size", type=int, choices=[16, 32, 64, 128], default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, choices=[10, 50, 100, 200], default=10, help="Number of training epochs.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam", help="Optimizer to use.")
    parser.add_argument("--learning_rate", type=float, choices=[0.001, 0.0001, 0.00001],
                        default=0.001, help="Learning rate.")
    parser.add_argument("--patience", type=int, choices=[10, 20, 50, 100, 200], default=50, help="Patience for early stopping.")
    parser.add_argument("--early_stop_delta", type=float, choices=[0.0, 0.01, 0.001, 0.0001], default=0.0001,
                        help="Minimum change in validation loss to qualify as an improvement for early stopping.")
    parser.add_argument("--weight_decay", type=float, choices=[0.0, 1e-5, 1e-4, 1e-3], default=0.0,
                        help="L2 regularization term (default: 0.0).")
    parser.add_argument("--lookback_window", type=int, choices=[10, 20, 30, 60], default=30,
                        help="Lookback window size.")
    parser.add_argument("--forecast_horizon", type=int, choices=[1, 7, 14, 30], default=7,
                        help="Forecast horizon.")

    # Transformer-specific parameters
    parser.add_argument("--d_model", type=int, choices=[64, 128, 256, 512], default=512, help="Model dimensionality.")
    parser.add_argument("--n_heads", type=int, choices=[4, 8, 16], default=8, help="Number of attention heads.")
    parser.add_argument("--d_ff", type=int, choices=[128, 256, 512, 1024], default=2048, help="Feed-forward dimensionality.")
    parser.add_argument("--num_encoder_layers", type=int, choices=[2, 4, 6, 8], default=6, help="Number of encoder layers.")
    parser.add_argument("--num_decoder_layers", type=int, choices=[2, 4, 6, 8], default=6, help="Number of decoder layers.")
    parser.add_argument("--dropout", type=float, choices=[0.1, 0.2, 0.3, 0.5], default=0.1, help="Dropout probability.")

    # Computation device
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], default="cuda", help="Device to use.")

    # Event type for logging
    parser.add_argument("--event", type=str, choices=["training", "testing", "hyperparam"], default="training",
                        help="Event type for logging.")

    return parser.parse_args()