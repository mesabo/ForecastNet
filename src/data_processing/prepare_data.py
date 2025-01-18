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

"""
Preprocess multivariate time-series data for training and evaluation.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# Add the `src` directory to `PYTHONPATH`
PROJECT_ROOT = Path(__file__).parent / "src"
sys.path.append(str(PROJECT_ROOT))

def preprocess_time_series_data(args):
    """
    Preprocesses raw time-series data and splits it into train, validation, and test sets.
    Generates sliding window datasets for each split.
    """
    try:
        # Define the project root directory
        project_root = Path(__file__).resolve().parents[2]
        raw_data_path = project_root / "data/raw/time_series_data.csv"

        # Ensure output directory exists
        processed_dir = project_root / (args.data_path or "data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(raw_data_path, parse_dates=["date"])
        scaler = MinMaxScaler()
        feature_columns = ["room1", "room2", "room3", "room4", "conso"]
        scaled_data = scaler.fit_transform(data[feature_columns])
        normalized_df = pd.DataFrame(scaled_data, columns=feature_columns)
        normalized_df["date"] = data["date"]

        train_split = int(len(normalized_df) * 0.7)
        val_split = int(len(normalized_df) * 0.85)
        train_data = normalized_df.iloc[:train_split]
        val_data = normalized_df.iloc[train_split:val_split]
        test_data = normalized_df.iloc[val_split:]

        train_data.to_csv(processed_dir / "train.csv", index=False)
        val_data.to_csv(processed_dir / "val.csv", index=False)
        test_data.to_csv(processed_dir / "test.csv", index=False)

        args.logger.info("Preprocessing complete. Train/Val/Test splits saved.")

        create_sliding_window_data(train_data, args.lookback_window, args.forecast_horizon,
                                   processed_dir / "train_sliding.csv")
        args.logger.info("Creating sliding window datasets for train split complete.")
        create_sliding_window_data(val_data, args.lookback_window, args.forecast_horizon,
                                   processed_dir / "val_sliding.csv")
        args.logger.info("Creating sliding window datasets for validation split complete.")
        create_sliding_window_data(test_data, args.lookback_window, args.forecast_horizon,
                                   processed_dir / "test_sliding.csv")
        args.logger.info("Creating sliding window datasets for test split complete.")
    except Exception as e:
        args.logger.error(f"Error during preprocessing: {traceback.format_exc()}")
        raise


def create_sliding_window_data(data, lookback, forecast, output_path):
    """
    Converts time-series data into sliding window format.

    Args:
        data (pd.DataFrame): Input time-series data.
        lookback (int): Number of past time steps used as input.
        forecast (int): Number of future time steps to predict as output.
        output_path (str): Path to save the sliding window formatted dataset.

    Returns:
        None: Saves the sliding window dataset to a CSV file.
    """
    features = data.drop(columns=["date", "conso"]).values  # Exclude date column
    target = data["conso"].values  # Use "conso" as the target
    x, y = [], []

    for i in range(len(features) - lookback - forecast + 1):
        # Collect input features for the lookback period
        x.append(features[i: i + lookback])  # Shape: (lookback, num_features)

        # Collect target values for the forecast horizon
        y.append(target[i + lookback: i + lookback + forecast])  # Shape: (forecast,)

    # Verify shapes of X and y
    print(f"First X shape: {np.array(x).shape}, First y shape: {np.array(y).shape}")

    # Save sliding window dataset to a CSV file
    sliding_df = pd.DataFrame({
        "x": [xi.tolist() for xi in x],  # Convert arrays to lists for storage
        "y": [yi.tolist() for yi in y],
    })

    try:
        sliding_df.to_csv(output_path, index=False)
    except Exception as e:
        raise Exception(f"Failed to save sliding window data to {output_path}. Error: {e}")


class TimeSeriesDataset(Dataset):
    """
    Dataset class for handling sliding window time-series data.

    Args:
        csv_path (str): Path to the sliding window formatted CSV file.
        lookback_window (int): Number of past time steps used as input.
        forecast_horizon (int): Number of future time steps to predict as output.
    """
    def __init__(self, csv_path, lookback_window, forecast_horizon):
        data = pd.read_csv(csv_path)
        self.x = np.array([np.array(eval(xi)) for xi in data["x"]])  # Shape: (num_samples, lookback, num_features)
        self.y = np.array([np.array(eval(yi)) for yi in data["y"]])  # Shape: (num_samples, forecast)

        # Ensure that y has the correct dimensions
        if len(self.y.shape) == 1:  # If y is (samples,), reshape to (samples, forecast_horizon)
            self.y = self.y[:, np.newaxis]

        # Validate shapes
        if len(self.x.shape) != 3:
            raise ValueError(f"X should have 3 dimensions: (samples, lookback, features). Found {self.x.shape}.")
        if len(self.y.shape) != 2:
            raise ValueError(f"Y should have 2 dimensions: (samples, forecast). Found {self.y.shape}.")

        # Verify lookback_window and forecast_horizon
        if self.x.shape[1] != lookback_window:
            raise ValueError(
                f"Mismatch in lookback_window. Expected {lookback_window}, got {self.x.shape[1]}."
            )
        if self.y.shape[1] != forecast_horizon:
            raise ValueError(
                f"Mismatch in forecast_horizon. Expected {forecast_horizon}, got {self.y.shape[1]}."
            )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)