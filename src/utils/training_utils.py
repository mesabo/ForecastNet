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

from pathlib import Path
import time
import torch
import psutil
import os
import numpy as np
import torch.optim as optim
import json


def load_predictions_and_targets(preds_path):
    """
    Loads predictions and targets from a saved .npz file.

    Args:
        preds_path (str or Path): Path to the .npz file containing predictions and targets.

    Returns:
        tuple: A tuple containing:
            - predictions (np.ndarray): Predicted values, shape (samples, steps).
            - targets (np.ndarray): Actual values, shape (samples, steps).
    """
    preds_path = Path(preds_path)

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {preds_path}")

    data = np.load(preds_path)
    predictions = data["predictions"]
    targets = data["targets"]

    print(f"Loaded predictions and targets from {preds_path}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    return predictions, targets


class TrainingProfiler:
    """
    A utility class to measure elapsed time and memory utilization during training or evaluation.
    """

    def __init__(self, save_path=None):
        self.start_time = None
        self.end_time = None
        self.max_memory = 0
        self.save_path = save_path

    def start(self):
        """
        Starts the timer and initializes memory tracking.
        """
        self.start_time = time.time()
        self.max_memory = 0

    def stop(self):
        """
        Stops the timer and calculates the elapsed time.
        """
        self.end_time = time.time()

    def record_memory(self):
        """
        Tracks the peak memory usage.
        """
        # Get the memory usage for the current process in MB
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / (1024 ** 2)
        self.max_memory = max(self.max_memory, current_memory)

    def elapsed_time(self):
        """
        Returns the elapsed time in seconds.
        """
        return self.end_time - self.start_time if self.end_time and self.start_time else None

    def report(self):
        """
        Returns a report of the elapsed time and peak memory usage.
        """
        return {
            "elapsed_time": self.elapsed_time(),
            "max_memory_mb": self.max_memory,
        }

    def save_report(self):
        """
        Saves the profiling report to a JSON file if a save path is provided.
        """
        if self.save_path:
            report = self.report()
            os.makedirs(Path(self.save_path).parent, exist_ok=True)
            with open(self.save_path, "w") as f:
                json.dump(report, f, indent=4)
            print(f"Profiler report saved to {self.save_path}")


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        path (str): Path to save the best model.
        verbose (bool): Whether to print messages about the early stopping process.
    """
    def __init__(self, patience=10, delta=0.0, path="checkpoint.pt", verbose=True):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_optimizer(model, optimizer_name, learning_rate, weight_decay=0):
    """
    Returns an optimizer for the model.

    Args:
        model (torch.nn.Module): The model to optimize.
        optimizer_name (str): Name of the optimizer. Options: ["adam", "sgd"].
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): L2 regularization term (default: 0).

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")