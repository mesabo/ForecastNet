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
    base_dir = Path("output/") / args.device / args.task / args.model
    event_type = args.event if hasattr(args, "event") else "training"

    # Create base directories
    paths = {
        "logs": base_dir / "logs" / f"{event_type}" / f"batch{args.batch_size}" /
                f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}.log",
        "models": base_dir / "models" / f"{event_type}" / f"batch{args.batch_size}" /
                  f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}.pth",
        "results": base_dir / "results" / f"{event_type}" / f"batch{args.batch_size}" /
                   f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}.csv",
        "metrics": base_dir / "metrics" / f"{event_type}" / f"batch{args.batch_size}" /
                   f"epoch{args.epochs}_lookback{args.lookback_window}_forecast{args.forecast_horizon}.csv",
    }

    # Ensure directories exist
    for key, path in paths.items():
        os.makedirs(path.parent, exist_ok=True)

    return paths