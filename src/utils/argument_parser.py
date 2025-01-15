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
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, choices=[0.001, 0.0001, 0.00001],
                        default=0.001, help="Learning rate.")
    parser.add_argument("--lookback_window", type=int, choices=[10, 20, 30, 60], default=30,
                        help="Lookback window size.")
    parser.add_argument("--forecast_horizon", type=int, choices=[1, 7, 14, 30], default=7,
                        help="Forecast horizon.")

    # Computation device
    parser.add_argument("--device", type=str, choices=["cuda", "mps", "cpu"], default="cuda", help="Device to use.")

    # Event type for logging
    parser.add_argument("--event", type=str, choices=["training", "hyperparam", "ablation"], default="training",
                        help="Event type for logging.")

    return parser.parse_args()
