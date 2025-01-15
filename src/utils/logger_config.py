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

import logging
from src.utils.output_config import get_output_paths

def setup_logger(args):
    """
    Configures and returns a logger for the application.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Get output paths for logs
    output_paths = get_output_paths(args)
    log_file_path = output_paths["logs"]

    # Configure the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Formatter for log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler for writing logs to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Logs will be saved to {log_file_path}")

    return logger