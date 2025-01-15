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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess multivariate time-series data for training and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from pathlib import Path
import traceback

# Add the `src` directory to `PYTHONPATH`
PROJECT_ROOT = Path(__file__).parent / "src"
sys.path.append(str(PROJECT_ROOT))


def preprocess_time_series_data(args):
    """
    Preprocesses the raw time-series data for training and evaluation.

    Args:
        args (argparse.Namespace): Command-line arguments including paths and configurations.
    """
    try:
        # Define the project root directory
        project_root = Path(__file__).resolve().parents[2]

        # Resolve the raw data path
        raw_data_path = project_root / (args.data_path or "data/raw/time_series_data.csv")
        data = pd.read_csv(raw_data_path, parse_dates=["date"])

        # Ensure output directory exists
        processed_dir = project_root / "data/processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Normalize data
        scaler = MinMaxScaler()
        feature_columns = ["room1", "room2", "room3", "room4", "conso"]
        scaled_data = scaler.fit_transform(data[feature_columns])
        normalized_df = pd.DataFrame(scaled_data, columns=feature_columns)
        normalized_df["date"] = data["date"]

        # Split data
        train_split = int(len(normalized_df) * 0.7)
        val_split = int(len(normalized_df) * 0.85)
        train_data = normalized_df.iloc[:train_split]
        val_data = normalized_df.iloc[train_split:val_split]
        test_data = normalized_df.iloc[val_split:]

        # Save splits
        train_data.to_csv(processed_dir / "train.csv", index=False)
        val_data.to_csv(processed_dir / "val.csv", index=False)
        test_data.to_csv(processed_dir / "test.csv", index=False)

        args.logger.info("Data split and saved successfully.")
        args.logger.info(f"Train data: {train_data.shape}, Val data: {val_data.shape}, Test data: {test_data.shape}")

        # Generate sliding window datasets
        create_sliding_window_data(
            train_data, args.lookback_window, args.forecast_horizon, processed_dir / "train_sliding.csv"
        )
        create_sliding_window_data(
            val_data, args.lookback_window, args.forecast_horizon, processed_dir / "val_sliding.csv"
        )
        create_sliding_window_data(
            test_data, args.lookback_window, args.forecast_horizon, processed_dir / "test_sliding.csv"
        )
    except Exception as e:
        args.logger.error(f"\n{10 * '‼️'}\nError during preprocessing\nDetails:\n{traceback.format_exc()}\n{10 * '‼️'}")
        raise

def create_sliding_window_data(data, lookback, forecast, output_path):
    """
    Converts time-series data into sliding window format.

    Args:
        data (pd.DataFrame): Input time-series data.
        lookback (int): Number of past days to consider as input.
        forecast (int): Number of future days to predict as output.
        output_path (str): Path to save the sliding-window formatted dataset.

    Returns:
        None: Saves the formatted dataset to a CSV file.
    """
    features = data.drop(columns=["date"]).values
    dates = data["date"].values
    x, y, x_dates, y_dates = [], [], [], []

    for i in range(len(features) - lookback - forecast + 1):
        x.append(features[i : i + lookback])
        y.append(features[i + lookback : i + lookback + forecast])
        x_dates.append(dates[i : i + lookback])
        y_dates.append(dates[i + lookback : i + lookback + forecast])

    sliding_df = pd.DataFrame(
        {
            "x": [np.array(xi).tolist() for xi in x],
            "y": [np.array(yi).tolist() for yi in y],
            "x_dates": [np.array(xd).tolist() for xd in x_dates],
            "y_dates": [np.array(yd).tolist() for yd in y_dates],
        }
    )
    try:
        sliding_df.to_csv(output_path, index=False)
    except Exception as e:
        raise Exception(f"Failed to save sliding window data to {output_path}. Error: {e}")