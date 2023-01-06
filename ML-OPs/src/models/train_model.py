import argparse

import pdb

import matplotlib.pyplot as plt
import torch

## the following script is in src/models/train_model.py
## the code can not find src fix this
from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel
import numpy as np


def training() -> None:
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=1e-3)
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MyAwesomeModel()
    model = model.to(device)

    train_set = mnist(train=True, in_folder="data/raw/corruptmnist", out_folder="data/processed")
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = 5
    epoch_loss = []
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x.to(device))
            loss = criterion(preds, y.to(device))
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
        print(np.mean(loss_tracker))
        epoch_loss.append(np.mean(loss_tracker))
    torch.save(model.state_dict(), "models/trained_model.pt")

    plt.figure()
    plt.plot(loss_tracker, "-")
    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.savefig(f"reports/figures/training_curve.png")

    plt.figure()
    plt.plot(epoch_loss, "-")
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.savefig(f"reports/figures/training_curve_epochs.png")


if __name__ == "__main__":
    training()