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

import torch
import logging

logger = logging.getLogger(__name__)

def setup_device(args):
    """
    Determines the computation device to use based on user input or auto-detection.

    Args:
        args (argparse.Namespace): Parsed arguments, including the `device` argument.

    Returns:
        torch.device: Selected device.
    """
    # User-specified device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device.index)}")
    elif args.device == "mps" and torch.backends.mps.is_available():
        try:
            device = torch.device('mps')
            logger.info("Using MPS device for computation.")
        except RuntimeError as e:
            logger.warning("MPS backend is available but the device could not be used.")
            logger.warning(f"Error: {e}")
            device = torch.device('cpu')
    elif args.device == "cpu":
        device = torch.device('cpu')
        logger.info("Using CPU device for computation.")
    elif args.device == "auto":
        # Auto-detect the best device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(device.index)}")
        elif torch.backends.mps.is_available():
            try:
                device = torch.device('mps')
                logger.info("Auto-detected MPS device for computation.")
            except RuntimeError as e:
                logger.warning("MPS backend is available but the device could not be used.")
                logger.warning(f"Error: {e}")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            logger.info("Auto-detected CPU device for computation.")
    else:
        device = torch.device('cpu')
        logger.warning(f"Invalid device argument '{args.device}'. Defaulting to CPU.")

    return device