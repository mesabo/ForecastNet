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

import os
from pathlib import Path


def get_output_paths(args):
    """
    Generate paths for logs, models, results, and metrics based on configuration.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        dict: A dictionary containing paths for logs, models, results, and metrics.
    """
    # Determine event_type for paths other than models
    event_type = args.event if hasattr(args, "event") else "training"

    base_dir = Path("output/") / args.device / args.task / args.model

    paths = {
        "logs": base_dir / "logs" / f"{event_type}" / args.frequency / f"batch{args.batch_size}" /
                f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}.log",
        "profiles": base_dir / "profiles" / f"{event_type}" / args.frequency / f"batch{args.batch_size}" /
                    f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "models": base_dir / "models" / (
            "training" if args.test_case in ["train_transformer", "evaluate_transformer"] else event_type) /
                  args.frequency / f"batch{args.batch_size}" /
                  f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "params": base_dir / "params" / (
            "training" if args.test_case in ["train_transformer", "evaluate_transformer"] else event_type) /
                  args.frequency / f"batch{args.batch_size}" /
                  f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "results": base_dir / "results" / f"{event_type}" / args.frequency / f"batch{args.batch_size}" /
                   f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "metrics": base_dir / "metrics" / f"{event_type}" / args.frequency / f"batch{args.batch_size}" /
                   f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "visuals": base_dir / "visuals" / f"{event_type}" / args.frequency / f"batch{args.batch_size}" /
                   f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}",
        "predictions": base_dir / "predictions" / f"{event_type}" / args.frequency / f"batch{args.batch_size}" /
                       f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}",
    }

    # Ensure all directory paths exist
    for key, path in paths.items():
        if key == "predictions":
            # Ensure predictions path exists but avoid treating file paths as directories
            os.makedirs(path, exist_ok=True)
        elif path.suffix:  # If the path has a file extension
            os.makedirs(path.parent, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)

    return paths