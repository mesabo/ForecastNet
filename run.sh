#!/bin/bash
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --nodelist=ai-gpgpu14      # Node to run the job

# Load user environment
source ~/.bashrc
hostname

# Detect computation device
DEVICE="cpu"
if [[ -n "$CUDA_VISIBLE_DEVICES" && $(nvidia-smi | grep -c "GPU") -gt 0 ]]; then
    DEVICE="cuda"
elif [[ "$(uname -s)" == "Darwin" && $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
    DEVICE="mps"
fi

# Activate Conda environment
ENV_NAME="itransformers"
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    if ! command -v conda &>/dev/null; then
        echo "Error: Conda not found. Please install Conda and ensure it's in your PATH."
        exit 1
    fi
    source activate "$ENV_NAME" || { echo "Error: Could not activate Conda environment '$ENV_NAME'."; exit 1; }
fi

# Configurations with choices
TASKS=("time_series")
TEST_CASES=("generate_data")  # Add more cases as needed: "train_transformer", "evaluate_transformer"
DATASETS=("room_data")
MODELS=("transformer")
BATCH_SIZES=("16")
LOOKBACK_WINDOWS=("10")
FORECAST_HORIZONS=("7")
LEARNING_RATES=("0.001")
EPOCHS=("10")

# Loop through configurations
for TASK in "${TASKS[@]}"; do
  for TEST_CASE in "${TEST_CASES[@]}"; do
    for EPOCH in "${EPOCHS[@]}"; do
      for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
          for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
            for LOOKBACK_WINDOW in "${LOOKBACK_WINDOWS[@]}"; do
              for FORECAST_HORIZON in "${FORECAST_HORIZONS[@]}"; do
                for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                  # Display high-level execution details in the terminal
                  echo "Running Configuration:"
                  echo "  Task: $TASK"
                  echo "  Test Case: $TEST_CASE"
                  echo "  Dataset: $DATASET"
                  echo "  Model: $MODEL"
                  echo "  Batch Size: $BATCH_SIZE"
                  echo "  Epochs: $EPOCH"
                  echo "  Lookback Window: $LOOKBACK_WINDOW"
                  echo "  Forecast Horizon: $FORECAST_HORIZON"
                  echo "  Learning Rate: $LEARNING_RATE"
                  echo "  Device: $DEVICE"

                  # Execute the Python script
                  python -u main.py \
                    --task "$TASK" \
                    --test_case "$TEST_CASE" \
                    --model "$MODEL" \
                    --data_path "./data/processed/train.csv" \
                    --batch_size "$BATCH_SIZE" \
                    --epochs "$EPOCH" \
                    --learning_rate "$LEARNING_RATE" \
                    --lookback_window "$LOOKBACK_WINDOW" \
                    --forecast_horizon "$FORECAST_HORIZON" \
                    --device "$DEVICE" \
                    --event "training"

                  # Check execution status
                  if [[ $? -ne 0 ]]; then
                      echo "Error: Execution failed for TASK=$TASK, TEST_CASE=$TEST_CASE, MODEL=$MODEL, DATASET=$DATASET."
                      exit 1
                  fi

                  echo "Execution complete for TASK=$TASK, TEST_CASE=$TEST_CASE, MODEL=$MODEL, DATASET=$DATASET."
                  echo "--------------------------------------------------------------------------------"
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟 CONGRATULATIONS 🌟🌟🌟🌟🌟🌟🌟🌟🌟"