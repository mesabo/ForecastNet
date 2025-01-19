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
import pandas as pd
from pathlib import Path


def find_best_hyperparams(metrics_dir, prioritize_metrics=None, top_n=5):
    """
    Finds the best hyperparameter combinations based on prioritized metrics.

    Args:
        metrics_dir (str): Path to the directory containing metrics CSV files.
        prioritize_metrics (list): List of metrics to prioritize (e.g., ["MSE", "MAE"]).
        top_n (int): Number of top configurations to return.

    Returns:
        pd.DataFrame: Top hyperparameter combinations and their corresponding metrics.
    """
    if prioritize_metrics is None:
        prioritize_metrics = ["MSE", "MAE", "RMSE", "MAPE"]

    # Collect all metrics files under /testing/ directory only
    metrics_dir = Path(metrics_dir)
    metrics_files = [
        file
        for file in metrics_dir.rglob("*.csv")
        if "/testing/" in file.as_posix()
    ]

    if not metrics_files:
        print(f"No metrics files found under '/testing/' in {metrics_dir}.")
        return pd.DataFrame()

    # Load and combine all metrics
    results = []
    for metrics_file in metrics_files:
        try:
            metrics = pd.read_csv(metrics_file)
            # Extract hyperparameter configuration from the file path
            relative_path = metrics_file.relative_to(metrics_dir).as_posix()
            # Add path as metadata
            metrics["Path"] = relative_path
            results.append(metrics)
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")

    if not results:
        print("No valid metrics data found.")
        return pd.DataFrame()

    # Combine results into a single DataFrame
    combined_metrics = pd.concat(results, ignore_index=True)

    # Ensure all prioritize metrics exist in the data
    missing_metrics = [m for m in prioritize_metrics if m not in combined_metrics.columns]
    if missing_metrics:
        print(f"Missing metrics in data: {missing_metrics}")
        return combined_metrics

    # Sort based on prioritized metrics (lowest is better)
    combined_metrics = combined_metrics.sort_values(by=prioritize_metrics, ascending=True)

    # Return the top N configurations
    return combined_metrics.head(top_n)


if __name__ == "__main__":
    # Specify the path to the metrics directory
    metrics_directory = "../output/mps/time_series/transformer/metrics"

    # Call the analysis function
    top_results = find_best_hyperparams(metrics_directory, prioritize_metrics=["MSE", "MAE", "RMSE", "MAPE"], top_n=5)

    # Print the top results
    if not top_results.empty:
        print("Top Hyperparameter Configurations:")
        print(top_results)

        # Optionally, save the results to a CSV file
        top_results.to_csv(f"{metrics_directory}/best_hyperparams_testing.csv", index=False)