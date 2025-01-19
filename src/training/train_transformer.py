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
from src.models.transformer_model import TransformerModel
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
        Saves profiling data (time and memory) for each epoch.
        """
        profiles_path = Path(self.args.output_paths["profiles"])
        profiles_path.mkdir(parents=True, exist_ok=True)
        profiling_file = profiles_path / "profiling.json"

        if profiling_file.exists():
            with open(profiling_file, "r") as f:
                existing_data = json.load(f)
            existing_data.append(profiling_data)
        else:
            existing_data = [profiling_data]

        with open(profiling_file, "w") as f:
            json.dump(existing_data, f, indent=4)

    def save_hyperparameters(self):
        """
        Saves the model hyperparameters to a JSON file for evaluation.
        """
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
        param_path = self.args.output_paths["params"]
        with open(param_path, "w") as f:
            json.dump(hyperparameters, f)
        self.args.logger.info(f"Model hyperparameters saved to {param_path}")

    def save_epoch_results(self):
        """
        Saves epoch results to a CSV file.
        """
        results_df = pd.DataFrame(self.results_log)
        train_metrics_df = pd.DataFrame(self.results_log["train_metrics"]).add_prefix("train_")
        val_metrics_df = pd.DataFrame(self.results_log["val_metrics"]).add_prefix("val_")
        results_df = pd.concat([
            results_df.drop(["train_metrics", "val_metrics"], axis=1),
            train_metrics_df,
            val_metrics_df,
        ], axis=1)
        results_df.to_csv(self.args.output_paths["results"], index=False)
        self.args.logger.info(f"Epoch results saved to {self.args.output_paths['results']}")

    def save_final_metrics_summary(self):
        """
        Saves the final metrics summary to a CSV file.
        """
        metrics_summary = pd.DataFrame(self.results_log["val_metrics"]).mean().to_dict()
        pd.DataFrame([metrics_summary]).to_csv(self.args.output_paths["metrics"], index=False)
        self.args.logger.info(f"Final metrics summary saved to {self.args.output_paths['metrics']}")

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
            path=self.args.output_paths["models"],
            verbose=True,
        )

        # Profiling: Start tracking memory and time
        start_time = time.time()
        initial_memory = self._track_memory()

        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_metrics = self._train_one_epoch()
            val_loss, val_metrics = self.validate()

            self.results_log["epoch"].append(epoch)
            self.results_log["train_loss"].append(train_loss)
            self.results_log["val_loss"].append(val_loss)
            self.results_log["train_metrics"].append(train_metrics)
            self.results_log["val_metrics"].append(val_metrics)

            # Save profiling for the epoch
            epoch_profiling = {
                "epoch": epoch,
                "time_elapsed": round(time.time() - start_time, 2),
                "memory_used_mb": round(self._track_memory() - initial_memory, 2),
            }
            self._save_profiling(epoch_profiling)

            # Early stopping check
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                self.args.logger.info("Early stopping triggered. Exiting training loop.")
                break

            # Generate training visualizations
            plot_loss_curve(
                self.results_log["train_loss"],
                self.results_log["val_loss"],
                self.args.output_paths["visuals"] / "loss_curve.png",
            )
            plot_metrics_trend(
                self.results_log["train_metrics"],
                self.results_log["val_metrics"],
                metric_names=["MSE", "MAE"],
                output_path=self.args.output_paths["visuals"] / "metrics_trend.png",
            )
            plot_learning_rate_schedule(
                learning_rates, self.args.output_paths["visuals"] / "learning_rate_schedule.png"
            )

        # Finalize profiling
        total_time = time.time() - start_time
        memory_used = self._track_memory() - initial_memory
        self.args.logger.info(f"Training completed in {total_time:.2f} seconds using {memory_used:.2f} MB of memory.")

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
        model_path = self.args.output_paths["models"]
        param_path = self.args.output_paths["params"]
        preds_path = self.args.output_paths["models"].parent / "predictions.npz"

        if not model_path.exists():
            self.args.logger.error(f"Model file not found at {model_path}. Please train the model first.")
            return

        if not param_path.exists():
            self.args.logger.error(f"Hyperparameters file not found at {param_path}. Cannot initialize model.")
            return

        # Load hyperparameters
        with open(param_path, "r") as f:
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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.args.logger.info(f"Model loaded from {model_path}.")

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
        os.makedirs(preds_path.parent, exist_ok=True)
        np.savez_compressed(preds_path, predictions=all_preds, targets=all_targets)
        self.args.logger.info(f"Predictions and targets saved to {preds_path}")

        # Calculate and save metrics
        metrics = self.metrics.calculate_all(all_targets, all_preds)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.args.output_paths["metrics"], index=False)

        # Save visualization plots
        qty = 0.2
        plot_multi_step_predictions(all_targets, all_preds, self.args.output_paths["visuals"], qty)
        plot_aggregated_steps(all_targets, all_preds, self.args.output_paths["visuals"], qty)
        plot_error_heatmap(all_targets, all_preds, self.args.output_paths["visuals"], qty)
        plot_residuals(all_targets, all_preds, self.args.output_paths["visuals"], qty)

        # Log profiling information
        total_time = end_time - start_time
        memory_used = final_memory - initial_memory
        self.args.logger.info(f"Evaluation completed in {total_time:.2f} seconds using {memory_used:.2f} MB of memory.")
        self.args.logger.info(
            f"Metrics and visualizations saved to {self.args.output_paths['metrics']} and {self.args.output_paths['visuals']}")
