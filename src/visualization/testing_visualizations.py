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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_multi_step_predictions(actual, predicted, output_path, qty=0.25):
    """
    Plots multi-step forecasting predictions for a subset of the data.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        output_path (str): Directory path to save the plots.
        qty (float): Fraction of the total rows to plot (e.g., 0.25 for the first 25% of rows).
    """
    length = int(len(actual) * qty)  # Number of rows to include in the plot
    n_steps = actual.shape[1]

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    for step in range(n_steps):
        plt.figure(figsize=(12, 6))
        plt.plot(actual[:length, step], label=f"Actual (Step {step + 1})", color="blue")
        plt.plot(predicted[:length, step], label=f"Predicted (Step {step + 1})", linestyle="dashed", color="orange")

        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.title(f"Multi-Step Forecasting Predictions (Step {step + 1})")
        plt.legend()
        plt.tight_layout()

        # Save each step's plot with a unique name
        step_output_path = os.path.join(output_path, str("predicted_vs_actual_step"), f"{step + 1}.png")
        os.makedirs(os.path.dirname(step_output_path), exist_ok=True)
        plt.savefig(step_output_path)
        plt.close()


def plot_aggregated_steps(actual, predicted, output_path, qty=0.25):
    """
    Plots aggregated multi-step forecasting predictions by flattening the arrays.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        output_path (str): Path to save the plot.
    """

    length = int(len(actual.flatten()) * qty)  # Number of rows to include in the plot
    plt.figure(figsize=(12, 6))
    plt.plot(actual.flatten()[:length], label="Actual", alpha=0.7)
    plt.plot(predicted.flatten()[:length], label="Predicted", alpha=0.7, linestyle="dashed")
    plt.xlabel("Flattened Samples")
    plt.ylabel("Values")
    plt.title("Aggregated Multi-Step Forecasting Predictions")
    plt.legend()
    plt.tight_layout()
    aggr_output_path = os.path.join(output_path, "aggregated_steps.png")
    os.makedirs(os.path.dirname(aggr_output_path), exist_ok=True)
    plt.savefig(aggr_output_path)
    plt.close()


def plot_error_heatmap(actual, predicted, output_path, qty=0.25):
    """
    Plots an error heatmap for multi-step forecasting predictions.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        output_path (str): Path to save the heatmap.
    """
    length = int(len(actual) * qty)  # Number of rows to include in the plot
    errors = np.abs(actual[:length] - predicted[:length])
    plt.figure(figsize=(12, 8))
    sns.heatmap(errors, annot=False, cmap="coolwarm", cbar=True)
    plt.title("Error Heatmap (Multi-Step Forecasting)")
    plt.xlabel("Forecast Steps")
    plt.ylabel("Samples")
    plt.tight_layout()
    ehm_output_path = os.path.join(output_path, "error_heatmap.png")
    os.makedirs(os.path.dirname(ehm_output_path), exist_ok=True)
    plt.savefig(ehm_output_path)
    plt.close()


def plot_residuals(actual, predicted, output_path, qty=0.25):
    """
    Plots residuals (actual - predicted) for each forecasting step.

    Args:
        actual (np.ndarray): Actual values, shape (samples, steps).
        predicted (np.ndarray): Predicted values, shape (samples, steps).
        output_path (str): Path to save the residual plots.
    """
    length = int(len(actual) * qty)  # Number of rows to include in the plot
    n_steps = actual.shape[1]
    fig, axes = plt.subplots(n_steps, 1, figsize=(12, 10), sharex=True)
    for step in range(n_steps):
        residuals = actual[:length, step] - predicted[:length, step]
        axes[step].plot(residuals, label=f"Residuals Step {step + 1}")
        axes[step].axhline(0, color="black", linestyle="--", alpha=0.7)
        axes[step].legend()

    plt.xlabel("Samples")
    plt.suptitle("Residuals Analysis (Multi-Step Forecasting)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    res_output_path = os.path.join(output_path, "residuals.png")
    os.makedirs(os.path.dirname(res_output_path), exist_ok=True)
    plt.savefig(res_output_path)
    plt.close()
