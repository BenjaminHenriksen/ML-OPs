import numpy as np
import torch


def mnist():
    """Loads MNIST dataset
    Returns:
        train_set: torch.utils.data.Dataset with training data
        test_set: torch.utils.data.Dataset with test data
    """
    content = []
    for i in range(4):
        content.append(np.load(f"data/corruptmnist/train_{i}.npz", allow_pickle=True))

    data_train = torch.tensor(np.concatenate([c["images"] for c in content])).reshape(-1, 1, 28, 28)
    targets_train = torch.tensor(np.concatenate([c["labels"] for c in content]))

    content = np.load(f"data/corruptmnist/test.npz", allow_pickle=True)
    data_test = torch.tensor(content["images"]).reshape(-1, 1, 28, 28)
    targets_test = torch.tensor(content["labels"])

    train_set = torch.utils.data.TensorDataset(data_train, targets_train)
    test_set = torch.utils.data.TensorDataset(data_test, targets_test)

    return train_set, test_set
