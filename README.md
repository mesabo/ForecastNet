
# ForecastNet: Multivariate Time Series Forecasting using Transformers

**ForecastNet** is a modular and scalable framework for multivariate time series forecasting using Transformer models. The project is designed to be reusable, testable, and adaptable for a wide range of applications, including energy consumption prediction, climate modeling, and more.

---

## Project Features

- **Custom Dataset Generation**: Supports synthetic data generation for experimentation.
- **Flexible Model Design**: Components of the Transformer architecture are separated for reusability and testing.
- **Scalable Training**: Optimized training pipelines with support for GPU, MPS, and CPU devices.
- **Visualization Tools**: Analyze and visualize predictions for better interpretability.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ForecastNet.git
   cd ForecastNet
   ```

2. Set up the environment using Conda:
   ```bash
   conda env create -f environment.yml
   conda activate forecastnet
   ```

3. Run the main script:
   ```bash
   python main.py --task time_series --model transformer --test_case train_transformer_seq2seq --data_path ./data/processed --save_path ./output/models --batch_size 32 --epochs 10
   ```

---

## Project Structure

```
.
├── README.md                   # Project overview
├── data
│   ├── raw                     # Raw data files
│   └── processed               # Preprocessed dataset files
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── environment.yml             # Conda environment file
├── main.py                     # Entry point to run the project
├── output
│   ├── logs                    # Training logs
│   └── models                  # Saved model checkpoints
│       └── transformer_best.pth
├── run.sh                      # Bash script to run training
├── tests                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_transformer.py
│   ├── test_training.py
│   └── test_visualization.py
└── src
    ├── __init__.py
    ├── data_processing         # Dataset generation and preparation
    │   ├── __init__.py
    │   ├── generate_data.py
    │   └── prepare_data.py
    ├── models                  # Transformer components and models
    │   ├── __init__.py
    │   ├── transformer_encoder.py
    │   ├── transformer_decoder.py
    │   ├── positional_encoding.py
    │   ├── multi_head_attention.py
    │   ├── feed_forward.py
    │   └── transformer_model.py
    ├── training                # Training and validation logic
    │   ├── __init__.py
    │   └── train_transformer.py
    ├── utils                   # Helper functions
    │   ├── __init__.py
    │   ├── argument_parser.py
    │   ├── device_utils.py
    │   ├── metrics.py
    │   └── training_utils.py
    └── visualization           # Visualization scripts
        ├── __init__.py
        └── plot_predictions.py
```

---

## Usage

### Generating Synthetic Data
Generate a synthetic dataset using:
```bash
python src/data_processing/generate_data.py
```

### Training the Model
Train the Transformer model with:
```bash
python main.py --task time_series --model transformer --test_case train_transformer_seq2seq --data_path ./data/processed --save_path ./output/models --batch_size 32 --epochs 10
```

### Visualizing Results
Use the visualization tools to analyze predictions:
```bash
python src/visualization/plot_predictions.py
```

---

## Contributing

We welcome contributions! Please fork the repository, make your changes, and submit a pull request.

---

## License

This project is licensed under the MIT License.

