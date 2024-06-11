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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import neps
from neps.plot.tensorboard_eval import tblogger


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super().__init__()
        layers = [nn.Flatten()]

        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            layers.append(nn.ReLU())
            input_size = num_neurons  # Set input size for the next layer

        layers.append(nn.Linear(num_neurons, 10))  # Output layer for 10 classes
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def training_pipeline(
        # neps parameters for load-save of checkpoints
        pipeline_directory,
        previous_pipeline_directory,
        # hyperparameters
        num_layers,
        num_neurons,
        learning_rate,
        optimizer,
        epochs
    ):
    _start = time.time()
    # Transformations applied on each image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Mean and Std Deviation for MNIST
        ]
    )

    start = time.time()
    # Loading MNIST dataset
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)
    data_load_time = time.time() - start

    model = SimpleNN(28 * 28, num_layers, num_neurons)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    optimizer_name = optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise KeyError(f"optimizer {optimizer} is not available")

    # Load possible checkpoint
    steps = None
    if previous_pipeline_directory is not None:
        checkpoint = torch.load(previous_pipeline_directory / "checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "steps" in checkpoint:
            steps = checkpoint["steps"]
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])

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
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        val_loss /= len(val_loader.dataset)

        # refer https://automl.github.io/neps/latest/examples/convenience/neps_tblogger_tutorial/
        start = time.time()
        tblogger.log(
            loss=np.mean(loss_per_batch),
            current_epoch=epoch+1,
            # write_summary_incumbent=True,  # this fails, need to fix?
            writer_config_scalar=True,
            writer_config_hparam=True,
            extra_data={"val_loss": tblogger.scalar_logging(value=val_loss)},
        )
        validation_time += (time.time() - start)
    training_time = time.time() - start - validation_time

    # Save checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "steps": epoch,
        },
        pipeline_directory / "checkpoint.pth",
    )

    return {
        "loss": val_loss,  # validation loss in the last epoch
        "cost": time.time() - _start,
        "info_dict": {
            "training_loss": np.mean(loss_per_batch),  # training loss in the last epoch
            "data_load_time": data_load_time,
            "training_time": training_time,
            "validation_time": validation_time,
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
