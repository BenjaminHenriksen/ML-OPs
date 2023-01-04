import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
import numpy as np
from matplotlib import pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = mnist()
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = 10
    epoch_loss = []
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")
        print(np.mean(loss_tracker))
        epoch_loss.append(np.mean(loss_tracker))

    plt.figure()
    plt.plot(epoch_loss, "-")
    plt.xlabel("Epochs")
    plt.ylabel("Training loss")
    plt.show()

    ## save model to use in evaluation
    torch.save(model.state_dict(), "trained_model.pt")





@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    ## call in terminal: python main.py evaluate trained_model.pt
    ## load the model that is saved as trained_model.pt
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    _, test_set = mnist()

    dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
    criterion = torch.nn.CrossEntropyLoss()

    ## test the accuracy of the model
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            preds = model(x)
            _, predicted = torch.max(preds.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    print(f"Accuracy of the network on the 10000 test images: {100 * correct / total} %")






cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  