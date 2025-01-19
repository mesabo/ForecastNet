# Time-Series Forecasting with Transformers

## Overview
This repository provides a comprehensive framework for multivariate time-series forecasting using Transformer-based models. The implementation includes data preprocessing, model training, evaluation, and visualization tools.

## Project Structure

```
.
├── README.md
├── data
│   ├── processed
│   │   ├── lookback30_forecast14
│   │   │   ├── test.csv
│   │   │   ├── test_sliding.csv
│   │   │   ├── train.csv
│   │   │   ├── train_sliding.csv
│   │   │   ├── val.csv
│   │   │   └── val_sliding.csv
│   │   ├── lookback30_forecast7
│   │   │   ├── test.csv
│   │   │   ├── test_sliding.csv
│   │   │   ├── train.csv
│   │   │   ├── train_sliding.csv
│   │   │   ├── val.csv
│   │   │   └── val_sliding.csv
│   │   ├── lookback60_forecast14
│   │   │   ├── test.csv
│   │   │   ├── test_sliding.csv
│   │   │   ├── train.csv
│   │   │   ├── train_sliding.csv
│   │   │   ├── val.csv
│   │   │   └── val_sliding.csv
│   │   └── lookback60_forecast7
│   │       ├── test.csv
│   │       ├── test_sliding.csv
│   │       ├── train.csv
│   │       ├── train_sliding.csv
│   │       ├── val.csv
│   │       └── val_sliding.csv
│   └── raw
│       └── time_series_data.csv
├── find_top_result.py
├── forecastnet_env.yml
├── main.py
├── output
│   └── mps
│       └── time_series
│           └── transformer
│               ├── logs
│               │   ├── testing
│               │   │   └── batch16
│               │   │       └── epoch50_lookback60_forecast7.log
│               │   └── training
│               │       └── batch16
│               │           └── epoch50_lookback60_forecast7.log
│               ├── metrics
│               │   ├── testing
│               │   │   └── batch16
│               │   │       └── epoch50_lookback60_forecast7
│               │   │           └── adam_0.0001_0.001.csv
│               │   └── training
│               │       └── batch16
│               │           └── epoch50_lookback60_forecast7
│               │               └── adam_0.0001_0.001.csv
│               ├── models
│               │   └── training
│               │       └── batch16
│               │           └── epoch50_lookback60_forecast7
│               │               └── adam_0.0001_0.001.pth
│               ├── params
│               │   └── training
│               │       └── batch16
│               │           └── epoch50_lookback60_forecast7
│               │               └── adam_0.0001_0.001.json
│               ├── predictions
│               │   ├── testing
│               │   │   └── batch16
│               │   │       └── epoch50_lookback60_forecast7
│               │   │           └── adam_0.0001_0.001.npz
│               │   └── training
│               │       └── batch16
│               │           └── epoch50_lookback60_forecast7
│               ├── profiles
│               │   ├── testing
│               │   │   └── batch16
│               │   │       └── epoch50_lookback60_forecast7
│               │   │           └── adam_0.0001_0.001.csv
│               │   └── training
│               │       └── batch16
│               │           └── epoch50_lookback60_forecast7
│               │               └── adam_0.0001_0.001.csv
│               ├── results
│               │   ├── testing
│               │   │   └── batch16
│               │   │       └── epoch50_lookback60_forecast7
│               │   └── training
│               │       └── batch16
│               │           └── epoch50_lookback60_forecast7
│               │               └── adam_0.0001_0.001.csv
│               └── visuals
│                   ├── testing
│                   │   └── batch16
│                   │       └── epoch50_lookback60_forecast7
│                   │           └── adam_0.0001_0.001
│                   │               ├── aggregated_steps.png
│                   │               ├── error_heatmap.png
│                   │               ├── multi_step_predictions.png
│                   │               ├── predicted_vs_actual_step
│                   │               │   ├── 1.png
│                   │               │   ├── 2.png
│                   │               │   ├── 3.png
│                   │               │   ├── 4.png
│                   │               │   ├── 5.png
│                   │               │   ├── 6.png
│                   │               │   └── 7.png
│                   │               └── residuals.png
│                   └── training
│                       └── batch16
│                           └── epoch50_lookback60_forecast7
│                               └── adam_0.0001_0.001
│                                   ├── learning_rate_schedule.png
│                                   ├── loss_curve.png
│                                   └── metrics_trend.png
├── run.sh
├── src
│   ├── data_processing
│   │   ├── generate_data.py
│   │   └── prepare_data.py
│   ├── models
│   │   └── transformer
│   │       ├── feed_forward.py
│   │       ├── multi_head_attention.py
│   │       ├── positional_encoding.py
│   │       ├── transformer_decoder.py
│   │       ├── transformer_encoder.py
│   │       └── transformer_model.py
│   ├── training
│   │   └── train_transformer.py
│   ├── utils
│   │   ├── argument_parser.py
│   │   ├── device_utils.py
│   │   ├── logger_config.py
│   │   ├── metrics.py
│   │   ├── output_config.py
│   │   └── training_utils.py
│   └── visualization
│       ├── attention_visualizations.py
│       ├── testing_visualizations.py
│       └── training_visualizations.py
└── tests
    ├── test_data_processing.py
    ├── test_training.py
    ├── test_transformer.py
    └── test_visualization.py
```

## Features

1. **Data Preprocessing**: 
   - Includes sliding window generation and dataset preparation.
   - Input data is stored in `data/processed/lookbackX_forecastY/`.

2. **Transformer Model**: 
   - Implemented in `src/models/transformer/`.
   - Includes attention mechanism, positional encoding, encoder, and decoder modules.

3. **Training and Evaluation**:
   - Handled by `src/training/train_transformer.py`.
   - Training results, metrics, and logs are saved to `output/mps/time_series/transformer/`.

4. **Visualization**:
   - Training and testing visualizations such as loss curves, metrics trends, and error heatmaps.
   - Output visualizations are stored under `output/mps/time_series/transformer/visuals/`.

5. **Hyperparameter Tuning**:
   - Results for different configurations are saved under `output/mps/time_series/transformer/metrics/`.
   - Use `find_top_result.py` to analyze and identify the best-performing configurations.

6. **Profiling**:
   - Tracks memory and time usage during training and evaluation.
   - Profiling results saved in `output/mps/time_series/transformer/profiles/`.

## Getting Started

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create the Conda environment:
   ```bash
   conda env create -f forecastnet_env.yml
   conda activate itransformers
   ```

### Data Preparation

1. Place raw time-series data in `data/raw/time_series_data.csv`.

2. Preprocess the data:
   ```bash
   python main.py --test_case preprocess_data
   ```

   Processed data will be saved under `data/processed/`.

### Model Training

1. Train the Transformer model:
   ```bash
   python main.py --test_case train_transformer
   ```

   Logs, metrics, models, and visualizations will be saved under `output/mps/time_series/transformer/`.

### Model Evaluation

1. Evaluate the trained model:
   ```bash
   python main.py --test_case evaluate_transformer
   ```

   Predictions, evaluation metrics, and visualizations will be saved under `output/mps/time_series/transformer/`.

### Hyperparameter Analysis

1. Find the best hyperparameter configurations:
   ```bash
   python find_top_result.py
   ```

   Top results will be printed and saved under `output/mps/time_series/transformer/metrics/`.

## Directory Structure

### Input Data
- **Raw Data**: `data/raw/time_series_data.csv`
- **Processed Data**: `data/processed/lookbackX_forecastY/`

### Outputs
- **Logs**: `output/mps/time_series/transformer/logs/`
- **Models**: `output/mps/time_series/transformer/models/`
- **Parameters**: `output/mps/time_series/transformer/params/`
- **Metrics**: `output/mps/time_series/transformer/metrics/`
- **Predictions**: `output/mps/time_series/transformer/predictions/`
- **Profiles**: `output/mps/time_series/transformer/profiles/`
- **Results**: `output/mps/time_series/transformer/results/`
- **Visualizations**: `output/mps/time_series/transformer/visuals/`

## Customization

1. Modify hyperparameters in `src/utils/argument_parser.py`.
2. Adjust the Transformer model components in `src/models/transformer/`.
3. Customize visualizations in `src/visualization/`.

## Testing

Run unit tests for the modules:
```bash
pytest tests/
```

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

---

## Author
- **Name**: Messou
- **Email**: mesabo18@gmail.com / messouaboya17@gmail.com
- **Github**: [https://github.com/mesabo](https://github.com/mesabo)
- **University**: Hosei University
- **Lab**: Prof. YU Keping's Lab

## License
This project is licensed under the MIT License.
