
# ForecastNet: A Multivariate Time-Series Forecasting Framework

## Project Overview
ForecastNet is a framework designed for multivariate time-series forecasting using Transformers. The project incorporates modularity, scalability, and reusability, enabling researchers and practitioners to efficiently train and evaluate models on various time-series datasets.

---

## Directory Structure

```
.
├── README.md
├── data
│   ├── processed
│   └── raw
│       └── time_series_data.csv
├── forecastnet_env.yml
├── main.py
├── output
│   └── <device>
│       └── <task>
│           └── <model>
│               ├── logs
│               │   └── <event_type>
│               │       └── batch<size>
│               │           └── epoch<epochs>_lookback<lookback>_forecast<forecast>.log
│               ├── metrics
│               │   └── <event_type>
│               │       └── batch<size>
│               │           └── epoch<epochs>_lookback<lookback>_forecast<forecast>.csv
│               ├── models
│               │   └── <event_type>
│               │       └── batch<size>
│               │           └── epoch<epochs>_lookback<lookback>_forecast<forecast>.pth
│               └── results
│                   └── <event_type>
│                       └── batch<size>
│                           └── epoch<epochs>_lookback<lookback>_forecast<forecast>.csv
├── run.sh
├── src
│   ├── data_processing
│   │   ├── generate_data.py
│   │   └── prepare_data.py
│   ├── models
│   │   ├── feed_forward.py
│   │   ├── multi_head_attention.py
│   │   ├── positional_encoding.py
│   │   ├── transformer_decoder.py
│   │   ├── transformer_encoder.py
│   │   └── transformer_model.py
│   ├── training
│   │   └── __init__.py
│   ├── utils
│   │   ├── argument_parser.py
│   │   ├── device_utils.py
│   │   ├── logger_config.py
│   │   ├── metrics.py
│   │   ├── output_config.py
│   │   └── training_utils.py
│   └── visualization
│       └── plot_predictions.py
└── tests
    ├── test_data_processing.py
    ├── test_training.py
    ├── test_transformer.py
    └── test_visualization.py
```

---

## Features
- Modular design for Transformer-based time-series forecasting.
- Logs, models, results, and metrics are neatly organized by device, task, and model type.
- Built-in support for multivariate time-series data generation and processing.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/forecastnet.git
   cd forecastnet
   ```

2. Set up the environment:
   ```bash
   conda env create -f forecastnet_env.yml
   conda activate forecastnet
   ```

---

## Usage

### Running the Project
Execute the following command to train or evaluate a Transformer model:
```bash
bash run.sh
```

### Logging and Outputs
All outputs (logs, models, metrics, results) are saved under the `output` directory:
- Logs: `output/<device>/<task>/<model>/logs/...`
- Models: `output/<device>/<task>/<model>/models/...`
- Metrics: `output/<device>/<task>/<model>/metrics/...`
- Results: `output/<device>/<task>/<model>/results/...`

---

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

---

## Author
- **Name**: Messou
- **Email**: mesabo18@gmail.com / messouaboya17@gmail.com
- **Github**: [https://github.com/mesabo](https://github.com/mesabo)
- **University**: Hosei University
- **Lab**: Prof. YU Keping's Lab
