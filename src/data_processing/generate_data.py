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
import numpy as np
import pandas as pd

def generate_multivariate_time_series(args, output_path="data/raw/time_series_data.csv"):
    """
    Generates 3 years of synthetic daily multivariate time series data.

    Args:
        output_path (str): Path to save the generated dataset.
    """
    np.random.seed(42)
    days = 365 * 3
    date_range = pd.date_range(start="2020-01-01", periods=days, freq="D")

    data = {
        "date": date_range,
        "room1": np.random.normal(20, 2, days),
        "room2": np.random.normal(21, 2, days),
        "room3": np.random.normal(22, 2, days),
        "room4": np.random.normal(23, 2, days),
        "conso": np.random.normal(100, 10, days)
    }
    if not args.data_path:
        output_path = not args.data_path
    df = pd.DataFrame(data)
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        args.logger.error(f"\n{10*'‼️'}\n Error details: {e}\n{10*'‼️'}\n")
    args.logger.info(f"Dataset saved to {output_path}")
