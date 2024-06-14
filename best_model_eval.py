import argparse
import logging
from pathlib import Path

from neps.status.status import get_summary_dict

from hpo_target import training_pipeline
from utils import load_neps_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_directory", type=str, default="./neps_debug/", required=True
    )
    parser.add_argument(
        "--max_budget", default=None, type=int, help="Maximum epochs to train for evaluation"
    )
    parser.add_argument(
        "--output_path", default="./output", type=str
    )
    parser.add_argument(
        "--eval_from_scratch", action="store_true"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = get_args()

    best_config_info = get_summary_dict(args.root_directory)
    best_hyperparameters = best_config_info["best_config"]

    print("Training the best found configuration:")
    _inc = best_hyperparameters.copy()
    _inc.pop("epochs", None)
    print(_inc)

    if args.max_budget is not None and args.max_budget > best_hyperparameters["epochs"]:
        best_hyperparameters["epochs"] = args.max_budget
    best_config_path = (
        Path(args.root_directory) / "results" / f"config_{best_config_info['best_config_id']}"
    )


    # Model training
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    training_pipeline(
        previous_pipeline_directory=best_config_path if not args.eval_from_scratch else None,
        pipeline_directory=output_path,
        **best_hyperparameters,
        log_tensorboard=False,
        verbose=True,
    )
