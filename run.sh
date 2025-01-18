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
TEST_CASES=("train_transformer" "evaluate_transformer") # "preprocess_data" "hyperparam_transformer" "ablation_transformer"
DATASETS=("room_data")
MODELS=("transformer")
BATCH_SIZES=("16")
LOOKBACK_WINDOWS=("30")
FORECAST_HORIZONS=("7")
LEARNING_RATES=("0.001")
EPOCHS=("10")
D_MODELS=("128")
N_HEADS=("8")
D_FFS=("256")
ENCODER_LAYERS=("4")
DECODER_LAYERS=("4")
DROPOUTS=("0.1")

# Loop through configurations
for TASK in "${TASKS[@]}"; do
  for TEST_CASE in "${TEST_CASES[@]}"; do

    # Dynamically set EVENT based on TEST_CASE
    if [[ "$TEST_CASE" == "evaluate_transformer" ]]; then
      EVENT="testing"
    elif [[ "$TEST_CASE" == "hyperparam_transformer" ]]; then
      EVENT="hyperparam"
    elif [[ "$TEST_CASE" == "ablation_transformer" ]]; then
      EVENT="ablation"
    else
      EVENT="training"
    fi

    for EPOCH in "${EPOCHS[@]}"; do
      for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
          for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
            for LOOKBACK_WINDOW in "${LOOKBACK_WINDOWS[@]}"; do
              for FORECAST_HORIZON in "${FORECAST_HORIZONS[@]}"; do
                for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
                  for D_MODEL in "${D_MODELS[@]}"; do
                    for N_HEAD in "${N_HEADS[@]}"; do
                      for D_FF in "${D_FFS[@]}"; do
                        for ENCODER_LAYER in "${ENCODER_LAYERS[@]}"; do
                          for DECODER_LAYER in "${DECODER_LAYERS[@]}"; do
                            for DROPOUT in "${DROPOUTS[@]}"; do

                              # Display high-level execution details in the terminal
                              echo "Running Configuration:"
                              echo "  Task: $TASK"
                              echo "  Test Case: $TEST_CASE"
                              echo "  Event: $EVENT"
                              echo "  Dataset: $DATASET"
                              echo "  Model: $MODEL"
                              echo "  Batch Size: $BATCH_SIZE"
                              echo "  Epochs: $EPOCH"
                              echo "  Lookback Window: $LOOKBACK_WINDOW"
                              echo "  Forecast Horizon: $FORECAST_HORIZON"
                              echo "  Learning Rate: $LEARNING_RATE"
                              echo "  d_model: $D_MODEL"
                              echo "  n_heads: $N_HEAD"
                              echo "  d_ff: $D_FF"
                              echo "  Encoder Layers: $ENCODER_LAYER"
                              echo "  Decoder Layers: $DECODER_LAYER"
                              echo "  Dropout: $DROPOUT"
                              echo "  Device: $DEVICE"

                              # Execute the Python script
                              python -u main.py \
                                --task "$TASK" \
                                --test_case "$TEST_CASE" \
                                --model "$MODEL" \
                                --data_path "data/processed" \
                                --batch_size "$BATCH_SIZE" \
                                --epochs "$EPOCH" \
                                --learning_rate "$LEARNING_RATE" \
                                --lookback_window "$LOOKBACK_WINDOW" \
                                --forecast_horizon "$FORECAST_HORIZON" \
                                --d_model "$D_MODEL" \
                                --n_heads "$N_HEAD" \
                                --d_ff "$D_FF" \
                                --num_encoder_layers "$ENCODER_LAYER" \
                                --num_decoder_layers "$DECODER_LAYER" \
                                --dropout "$DROPOUT" \
                                --device "$DEVICE" \
                                --event "$EVENT"

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
          done
        done
      done
    done
  done
done

echo "🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟 CONGRATULATIONS 🌟🌟🌟🌟🌟🌟🌟🌟🌟"