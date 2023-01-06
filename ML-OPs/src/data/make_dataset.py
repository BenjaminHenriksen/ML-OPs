# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import Tuple
from torch import Tensor

class mnist(Dataset):
    def __init__(self, train: bool, in_folder: str = "", out_folder: str = "") -> None:
        super().__init__()

        self.in_folder = in_folder
        self.out_folder = out_folder
        self.train = train

        if self.train:
            content = []
            for i in range(4):
                content.append(np.load(f"{in_folder}/train_{i}.npz", allow_pickle=True))

            data = torch.tensor(np.concatenate([c["images"] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c["labels"] for c in content]))
        else:
            content = np.load(f"{in_folder}/test.npz", allow_pickle=True)
            data = torch.tensor(content["images"]).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content["labels"])

        self.data = data
        self.targets = targets

        if self.out_folder:
            split = "train" if self.train else "test"
            torch.save([self.data, self.targets], f"{self.out_folder}/{split}_processed.pt")

    def load_preprocessed(self) -> None:
        split = "train" if self.train else "test"
        self.data, self.targets = torch.load(f"{self.out_folder}/{split}_processed.pt")
    
    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx].float(), self.targets[idx]


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    train = mnist(train=True, in_folder=input_filepath, out_folder=output_filepath)
    train.save_preprocessed()

    test = mnist(train=False, in_folder=input_filepath, out_folder=output_filepath)
    test.save_preprocessed()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
