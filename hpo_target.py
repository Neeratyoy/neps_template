import argparse
import logging

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

    # Load possible checkpoint
    start = time.time()
    steps, model, optimizer = load_neps_checkpoint(previous_pipeline_directory, model, optimizer)
    checkpoint_load_time = time.time() - start

    start = time.time()
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
    training_time = time.time() - start - (
        validation_time + data_load_time + checkpoint_load_time + logging_time
    )

    # Save checkpoint
    save_neps_checkpoint(pipeline_directory, epoch, model, optimizer)

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo", type=str, default="priorband", choices=["priorband", "bo", "hyperband", "rs"]
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = get_args()

    match args.algo:
        case "priorband":
            run_args = "run_pb.yaml"
        case "bo":
            run_args = "run_bo.yaml"
        case "hyperband":
            run_args = "run_hb.yaml"
        case "rs":
            run_args = "run_rs.yaml"
        case _:
            raise ValueError(f"Invalid algo: {args.algo}")

    neps.run(run_args=Path(__file__).parent.absolute() / run_args)
