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
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_multi_step_predictions(actual, predicted, output_path):
    n_steps = actual.shape[1]
    plt.plot(actual[:, 0], label=f"Actual")
    for step in range(n_steps):
        plt.plot(predicted[:, step], label=f"Predicted Step {step + 1}", linestyle="dashed")

    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.title("Multi-Step Forecasting Predictions")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_aggregated_steps(actual, predicted, output_path):
    plt.plot(actual.flatten(), label="Actual", alpha=0.7)
    plt.plot(predicted.flatten(), label="Predicted", alpha=0.7, linestyle="dashed")
    plt.xlabel("Flattened Samples")
    plt.ylabel("Values")
    plt.title("Aggregated Multi-Step Forecasting Predictions")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_error_heatmap(actual, predicted, output_path):
    errors = np.abs(actual - predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(errors, annot=False, cmap="coolwarm", cbar=True)
    plt.title("Error Heatmap (Multi-Step Forecasting)")
    plt.xlabel("Forecast Steps")
    plt.ylabel("Samples")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_residuals(actual, predicted, output_path):
    n_steps = actual.shape[1]
    fig, axes = plt.subplots(n_steps, 1, figsize=(10, 8), sharex=True)
    for step in range(n_steps):
        residuals = actual[:, step] - predicted[:, step]
        axes[step].plot(residuals, label=f"Residuals Step {step + 1}")
        axes[step].axhline(0, color="black", linestyle="--", alpha=0.7)
        axes[step].legend()

    plt.xlabel("Samples")
    plt.suptitle("Residuals Analysis (Multi-Step Forecasting)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()