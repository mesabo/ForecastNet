import sys
from pathlib import Path

# Import test cases (to be implemented in subsequent steps)
from src.data_processing.generate_data import generate_multivariate_time_series
from src.data_processing.prepare_data import preprocess_time_series_data
from src.training.train_transformer import TrainTransformer
from src.utils.argument_parser import get_arguments
from src.utils.device_utils import setup_device
from src.utils.logger_config import setup_logger
from src.utils.output_config import get_output_paths

# Add the `src` directory to `PYTHONPATH`
PROJECT_ROOT = Path(__file__).parent / "src"
sys.path.append(str(PROJECT_ROOT))

# Map test cases to functions
TEST_CASES = {
    "generate_data": generate_multivariate_time_series,
    "preprocess_data": preprocess_time_series_data,
    "train_transformer": lambda args: TrainTransformer(args).train(),
    "evaluate_transformer": lambda args: TrainTransformer(args).evaluate(
        Path(args.data_path or "data/processed")
    ),
}

def main():
    # Parse arguments
    args = get_arguments()

    # Setup device
    device = setup_device(args)
    args.device = device.type  # Add device information to args

    # Get output paths
    output_paths = get_output_paths(args)

    # Setup logger
    logger = setup_logger(args)
    args.logger = logger
    args.output_paths = output_paths

    # Log the configuration details
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Lookback Window: {args.lookback_window}")
    logger.info(f"Forecast Horizon: {args.forecast_horizon}")
    logger.info(f"Log Path: {output_paths['logs']}")
    logger.info(f"Model Path: {output_paths['models']}")
    logger.info(f"Results Path: {output_paths['results']}")
    logger.info(f"Metrics Path: {output_paths['metrics']}")
    logger.info(f"Visuals Path: {output_paths['visuals']}")

    # Execute the selected test case
    if args.test_case in TEST_CASES:
        logger.info(f"{10 * '🌟'} Running {args.test_case} {10 * '🌟'}")
        TEST_CASES[args.test_case](args)  # Pass arguments to the function
    else:
        logger.error(f"Invalid test case: {args.test_case}")
        logger.error(f"Available test cases: {list(TEST_CASES.keys())}")

    logger.info(f"{10 * '🏁'} ALL EXECUTIONS DONE! {10 * '🏁'}")

if __name__ == "__main__":
    main()
