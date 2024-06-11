import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import neps


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
    ):
    epochs = 2

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

    model = SimpleNN(28 * 28, num_layers, num_neurons)
    criterion = nn.CrossEntropyLoss()

    model = model.to("mps")

    # Select optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise KeyError(f"optimizer {optimizer} is not available")

    # Training loop

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to("mps")
            target = target.to("mps")
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to("mps")
            target = target.to("mps")
            output = model(data)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader.dataset)
    # val_loss = val_loss.item()

    return val_loss


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neps.run(run_args="run_args.yaml")
    neps.run(
        run_pipeline=training_pipeline,
        pipeline_space=dict(
            num_layers=neps.IntegerParameter(1, 3),
            num_neurons=neps.IntegerParameter(8, 64),
            learning_rate=neps.FloatParameter(0.0001, 0.1, log=True),
            optimizer=neps.CategoricalParameter(["adam", "sgd"]),
        ),
        max_evaluations_total=10,
        searcher="random_search",
        root_directory="./neps_output/random_search/",
        post_run_summary=True,
    )
