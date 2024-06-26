import argparse
import logging

from multiprocessing import Process
import numpy as np
import os
from pathlib import Path
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim

import neps
from neps.plot.tensorboard_eval import tblogger

from utils import (
    load_neps_checkpoint,
    save_neps_checkpoint,
    prepare_mnist_dataloader,
    set_seeds,
    SimpleNN
)


def training_pipeline(
    # neps parameters for load-save of checkpoints
    pipeline_directory,
    previous_pipeline_directory,
    # hyperparameters
    batch_size,
    num_layers,
    num_neurons,
    learning_rate,
    weight_decay,
    optimizer,
    epochs,
    # other parameters
    log_tensorboard=True,
    verbose=False,
):
    # Load data
    _start = time.time()
    train_loader, val_loader = prepare_mnist_dataloader(batch_size=batch_size)
    data_load_time = time.time() - _start

    # Instantiate model and loss
    model = SimpleNN(28 * 28, num_layers, num_neurons)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    optimizer_name = optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise KeyError(f"optimizer {optimizer} is not available")

    # Initialize LR scheduler
    scheduler = None  # type: torch.optim.lr_scheduler.LRScheduler

    # Load possible checkpoint
    start = time.time()
    steps, model, optimizer = load_neps_checkpoint(previous_pipeline_directory, model, optimizer, scheduler)
    checkpoint_load_time = time.time() - start

    train_start = time.time()
    validation_time = 0
    # Training loop
    steps = steps or 0  # accounting for continuation if checkpoint loaded
    for epoch in range(steps, epochs):
        model.train()
        loss_per_batch = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_per_batch.append(loss.item())

        # perform validation per epoch
        start = time.time()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader.dataset)
        validation_time += (time.time() - start)

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"loss: {np.mean(loss_per_batch):.5f}, "
                f"val loss: {val_loss:.5f}"
            )

        # refer https://automl.github.io/neps/latest/examples/convenience/neps_tblogger_tutorial/
        start = time.time()
        if log_tensorboard:
            tblogger.log(
                loss=val_loss,
                current_epoch=epoch+1,
                # write_summary_incumbent=True,  # this fails, need to fix?
                writer_config_scalar=True,
                writer_config_hparam=True,
                extra_data={"train_loss": tblogger.scalar_logging(value=np.mean(loss_per_batch))},
            )
        logging_time = time.time() - start
    training_time = time.time() - train_start - validation_time - logging_time

    # Save checkpoint
    save_neps_checkpoint(pipeline_directory, epoch, model, optimizer, scheduler)

    return {
        "loss": val_loss,  # validation loss in the last epoch
        "cost": time.time() - _start,
        "info_dict": {
            "training_loss": np.mean(loss_per_batch),  # training loss in the last epoch
            "data_load_time": data_load_time,
            "training_time": training_time,
            "validation_time": validation_time,
            "checkpoint_load_time": checkpoint_load_time,
            "logging_time": logging_time,
            "hyperparameters": {
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,
                "epochs": epochs,
            },
        }
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        default="priorband",
    )    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    return args


def _worker(run_args_file):
    neps.run(run_args=run_args_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = get_args()

    # Setting file extension
    run_args_file = f"{args.algo}.yaml"

    # Setting full path
    run_args_file = Path(__file__).parent.absolute() / "configs" / run_args_file
    if not run_args_file.exists():
        raise ValueError(f"Invalid algo: {args.algo}. File {run_args_file} not found!")

    set_seeds(args.seed)
    if args.n_workers is None:
        _worker(run_args_file)
    else:
        processes = []
        # Start workers
        for _ in range(args.n_workers):
            p = Process(target=_worker, args=(run_args_file,))
            processes.append(p)
            p.start()
        # Wait for all workers to finish
        for p in processes:
            p.join()

