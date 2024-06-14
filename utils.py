import numpy as np
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import Tuple


def prepare_mnist_dataloader() -> Tuple[DataLoader, DataLoader]:
    # Transformations applied on each image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Mean and Std Deviation for MNIST
        ]
    )
    # Loading MNIST dataset
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_set, val_set = torch.utils.data.random_split(dataset, [55000, 5000])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)

    return train_loader, val_loader


def load_neps_checkpoint(
        previous_pipeline_directory: Path, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> Tuple[int, nn.Module, torch.optim.Optimizer]:
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
    return steps, model, optimizer


def save_neps_checkpoint(
    pipeline_directory: Path, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer
) -> None:
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
