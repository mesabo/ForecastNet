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

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_processing.prepare_data import TimeSeriesDataset
from src.models.transformer.transformer_model import TransformerModel
from src.utils.metrics import Metrics
from src.utils.training_utils import EarlyStopping, get_optimizer
from src.visualization.testing_visualizations import (
    plot_multi_step_predictions,
    plot_aggregated_steps,
    plot_error_heatmap,
    plot_residuals,
)
from src.visualization.training_visualizations import (
    plot_loss_curve,
    plot_metrics_trend,
    plot_learning_rate_schedule,
)


class TrainTransformer:
    """
    Handles the training and evaluation process for the Transformer model.
    """

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.metrics = Metrics(seasonality=args.forecast_horizon)

        # Placeholder for datasets and loaders
        self.train_loader = None
        self.val_loader = None

        # Initialize placeholders for input and output dimensions
        self.input_dim = None
        self.output_dim = args.forecast_horizon

        # Initialize results log
        self.results_log = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
        }
        self.criterion = nn.MSELoss()

        self.common_name = f"{args.optimizer}_{args.weight_decay}_{args.learning_rate}"
        self.model_path = self.args.output_paths["models"] / f"{self.common_name}.pth"
        self.param_path = self.args.output_paths["params"] / f"{self.common_name}.json"
        self.preds_path = self.args.output_paths["predictions"] / f"{self.common_name}.npz"
        self.visual_path = self.args.output_paths["visuals"] / self.common_name
        self.profile_path = self.args.output_paths["profiles"] / f"{self.common_name}.csv"
        self.result_path = self.args.output_paths["results"] / f"{self.common_name}.csv"
        self.metric_path = self.args.output_paths["metrics"] / f"{self.common_name}.csv"

    def load_data(self):
        """
        Loads the dataset and creates data loaders.
        """
        data_dir = Path(
            self.args.data_path) / f"lookback{self.args.lookback_window}_forecast{self.args.forecast_horizon}"
        train_dataset = TimeSeriesDataset(
            data_dir / "train_sliding.csv", self.args.lookback_window, self.args.forecast_horizon
        )
        val_dataset = TimeSeriesDataset(
            data_dir / "val_sliding.csv", self.args.lookback_window, self.args.forecast_horizon
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)

        self.input_dim = train_dataset.x.shape[2]

    def init_model(self):
        """
        Initializes the Transformer model using the dynamically determined input dimension.
        """
        max_len = max(self.args.lookback_window, self.args.forecast_horizon)

        self.model = TransformerModel(
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            d_ff=self.args.d_ff,
            num_encoder_layers=self.args.num_encoder_layers,
            num_decoder_layers=self.args.num_decoder_layers,
            dropout=self.args.dropout,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            max_len=max_len,
        ).to(self.device)

        # Correct call to get_optimizer
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _track_memory(self):
        """
        Tracks memory usage in MB.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def _save_profiling(self, profiling_data):
        """
        Saves profiling data (time and memory) for each epoch to a CSV file.
        """
        # Ensure the profiling directory exists

        # Include common_name in the profiling file name

        # Convert profiling data to a DataFrame for appending
        profiling_df = pd.DataFrame([profiling_data])

        # Check if the profiling file exists and append or create new
        if self.profile_path.exists():
            existing_data = pd.read_csv(self.profile_path)
            updated_data = pd.concat([existing_data, profiling_df], ignore_index=True)
        else:
            updated_data = profiling_df

        # Save updated profiling data to the CSV file
        updated_data.to_csv(self.profile_path, index=False)

        self.args.logger.info(f"Profiling data saved to {self.profile_path}")

    def save_hyperparameters(self):
        """
        Saves the model hyperparameters to a JSON file for evaluation.
        """
        # Ensure `self.common_name` is initialized in the constructor (__init__) as:
        # self.common_name = f"{args.optimizer}_{args.weight_decay}_{args.learning_rate}"

        hyperparameters = {
            "d_model": self.args.d_model,
            "n_heads": self.args.n_heads,
            "d_ff": self.args.d_ff,
            "num_encoder_layers": self.args.num_encoder_layers,
            "num_decoder_layers": self.args.num_decoder_layers,
            "dropout": self.args.dropout,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "max_len": max(self.args.lookback_window, self.args.forecast_horizon),
        }

        # Construct the parameter file path

        # Save hyperparameters to the file
        with open(self.param_path, "w") as f:
            json.dump(hyperparameters, f, indent=4)

        self.args.logger.info(f"Model hyperparameters saved to {self.param_path}")

    def save_epoch_results(self):
        """
        Saves epoch results to a CSV file, including the common_name in the filename.
        """
        # Convert the results log into DataFrames
        results_df = pd.DataFrame(self.results_log)
        train_metrics_df = pd.DataFrame(self.results_log["train_metrics"]).add_prefix("train_")
        val_metrics_df = pd.DataFrame(self.results_log["val_metrics"]).add_prefix("val_")

        # Combine results with metrics
        full_results_df = pd.concat([
            results_df.drop(["train_metrics", "val_metrics"], axis=1),
            train_metrics_df,
            val_metrics_df,
        ], axis=1)

        # Save to CSV
        os.makedirs(self.result_path.parent, exist_ok=True)

        full_results_df.to_csv(self.result_path, index=False)
        self.args.logger.info(f"Epoch results saved to {self.result_path}")

    def save_final_metrics_summary(self):
        """
        Saves the final metrics summary to a CSV file, including the common_name in the filename.
        """
        # Compute the average metrics summary
        metrics_summary = pd.DataFrame(self.results_log["val_metrics"]).mean().to_dict()

        # Save to CSV
        os.makedirs(self.metric_path.parent, exist_ok=True)
        pd.DataFrame([metrics_summary]).to_csv(self.metric_path, index=False)
        self.args.logger.info(f"Final metrics summary saved to {self.metric_path}")

    def train(self):
        """
        Runs the training loop for the Transformer model with profiling and early stopping.
        """
        self.load_data()
        self.init_model()

        best_val_loss = float("inf")
        learning_rates = []
        # Initialize EarlyStopping
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            delta=self.args.early_stop_delta,
            path=self.model_path,
            verbose=True,
        )

        # Profiling: Start tracking memory and time
        start_time = time.time()
        initial_memory = self._track_memory()

        profiling_records = []  # To store profiling records for the entire training

        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_metrics = self._train_one_epoch()
            val_loss, val_metrics = self.validate()

            self.results_log["epoch"].append(epoch)
            self.results_log["train_loss"].append(train_loss)
            self.results_log["val_loss"].append(val_loss)
            self.results_log["train_metrics"].append(train_metrics)
            self.results_log["val_metrics"].append(val_metrics)

            # Profiling for the epoch
            epoch_time = time.time() - start_time
            memory_used = self._track_memory() - initial_memory
            epoch_profiling = {
                "epoch": epoch,
                "time_elapsed": round(epoch_time, 2),
                "memory_used_mb": round(memory_used, 2),
            }
            profiling_records.append(epoch_profiling)

            # Early stopping check
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                self.args.logger.info("Early stopping triggered. Exiting training loop.")
                break

            # Generate training visualizations
            plot_loss_curve(
                self.results_log["train_loss"],
                self.results_log["val_loss"],
                self.visual_path / "loss_curve.png",
            )
            plot_metrics_trend(
                self.results_log["train_metrics"],
                self.results_log["val_metrics"],
                metric_names=["MSE", "MAE"],
                output_path=self.visual_path / "metrics_trend.png",
            )
            plot_learning_rate_schedule(
                learning_rates, self.visual_path / "learning_rate_schedule.png"
            )

        # Save all profiling data to a CSV file
        profiling_df = pd.DataFrame(profiling_records)
        profiling_df.to_csv(self.profile_path, index=False)
        self.args.logger.info(f"Training profiling data saved to {self.profile_path}")

        # Finalize profiling
        total_time = time.time() - start_time
        total_memory_used = self._track_memory() - initial_memory
        self.args.logger.info(
            f"Training completed in {total_time:.2f} seconds using {total_memory_used:.2f} MB of memory.")

        self.save_hyperparameters()
        self.save_epoch_results()
        self.save_final_metrics_summary()

    def _train_one_epoch(self):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        train_loss = 0.0
        all_preds, all_targets = [], []

        for x_batch, y_batch in self.train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x_batch, y_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(y_batch.detach().cpu().numpy())

        train_loss /= len(self.train_loader)
        train_metrics = self.metrics.calculate_all(np.concatenate(all_targets), np.concatenate(all_preds))
        return train_loss, train_metrics

    def validate(self):
        """
        Validates the model on the validation dataset.
        """
        self.model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(x_batch, y_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        val_loss /= len(self.val_loader)
        val_metrics = self.metrics.calculate_all(np.concatenate(all_targets), np.concatenate(all_preds))
        return val_loss, val_metrics

    def evaluate(self, test_path):
        """
        Evaluates the model on the test dataset with profiling.
        """
        test_path = (
                Path(test_path)
                / f"lookback{self.args.lookback_window}_forecast{self.args.forecast_horizon}"
                / "test_sliding.csv"
        )

        if not self.model_path.exists():
            self.args.logger.error(f"Model file not found at {self.model_path}. Please train the model first.")
            return

        if not self.param_path.exists():
            self.args.logger.error(f"Hyperparameters file not found at {self.param_path}. Cannot initialize model.")
            return

        # Load hyperparameters
        with open(self.param_path, "r") as f:
            hyperparameters = json.load(f)

        self.model = TransformerModel(
            d_model=hyperparameters["d_model"],
            n_heads=hyperparameters["n_heads"],
            d_ff=hyperparameters["d_ff"],
            num_encoder_layers=hyperparameters["num_encoder_layers"],
            num_decoder_layers=hyperparameters["num_decoder_layers"],
            dropout=hyperparameters["dropout"],
            input_dim=hyperparameters["input_dim"],
            output_dim=hyperparameters["output_dim"],
            max_len=hyperparameters["max_len"],
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.args.logger.info(f"Model loaded from {self.model_path}.")

        test_dataset = TimeSeriesDataset(test_path, self.args.lookback_window, self.args.forecast_horizon)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        self.model.eval()
        all_preds, all_targets = [], []

        # Profiling: Start tracking memory and time
        start_time = time.time()
        initial_memory = self._track_memory()

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                preds = self.model(x_batch, y_batch)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        # Profiling: End tracking memory and time
        end_time = time.time()
        final_memory = self._track_memory()

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Save predictions and targets
        os.makedirs(self.preds_path.parent, exist_ok=True)  # Ensure the parent directory exists
        np.savez_compressed(str(self.preds_path), predictions=all_preds, targets=all_targets)  # Save file
        self.args.logger.info(f"Predictions and targets saved to {self.preds_path}")

        # Calculate and save metrics
        metrics = self.metrics.calculate_all(all_targets, all_preds)

        os.makedirs(self.metric_path.parent, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(self.metric_path, index=False)
        self.args.logger.info(f"Metrics saved to {self.metric_path}")

        # Save visualization plots
        qty = 1
        plot_multi_step_predictions(
            all_targets,
            all_preds,
            self.visual_path / "multi_step_predictions.png",
            qty,
        )
        plot_aggregated_steps(
            all_targets,
            all_preds,
            self.visual_path / "aggregated_steps.png",
            qty,
        )
        plot_error_heatmap(
            all_targets,
            all_preds,
            self.visual_path / "error_heatmap.png",
            qty,
        )
        plot_residuals(
            all_targets,
            all_preds,
            self.visual_path / "residuals.png",
            qty,
        )

        # Save profiling information
        profiling_data = {
            "time_elapsed": round(end_time - start_time, 2),
            "memory_used_mb": round(final_memory - initial_memory, 2),
        }
        profiling_df = pd.DataFrame([profiling_data])
        profiling_df.to_csv(self.profile_path, index=False)

        self.args.logger.info(
            f"Evaluation completed in {profiling_data['time_elapsed']} seconds using {profiling_data['memory_used_mb']} MB of memory."
        )
        self.args.logger.info(f"Evaluation profiling saved to {self.profile_path}")
