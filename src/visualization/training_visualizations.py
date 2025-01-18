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

import matplotlib.pyplot as plt
import os

def plot_loss_curve(train_loss, val_loss, output_path):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_metrics_trend(train_metrics, val_metrics, metric_names, output_path):
    plt.figure()
    for metric_name in metric_names:
        plt.plot([m[metric_name] for m in train_metrics], label=f"Train {metric_name}")
        plt.plot([m[metric_name] for m in val_metrics], label=f"Val {metric_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Metrics Trend")
    plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_learning_rate_schedule(lr_schedule, output_path):
    plt.figure()
    plt.plot(lr_schedule)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()