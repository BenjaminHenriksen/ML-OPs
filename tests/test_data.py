from tests import _PATH_DATA
import os
import torch
from src.data.make_dataset import mnist
import pytest

#_PATH_DATA = os.path.join(_PATH_DATA, "datdsgsa")

def test_path_data():
    """Test if the data folder exists"""
    assert os.path.exists(_PATH_DATA)

## import the train and test data
if os.path.exists(_PATH_DATA):  
    train_data = mnist(train=True, in_folder=_PATH_DATA + "/raw", out_folder=_PATH_DATA + "/processed")
    test_data = mnist(train=False, in_folder=_PATH_DATA + "/raw", out_folder=_PATH_DATA + "/processed")

data_there = pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")

@data_there
def test_path_data():
    """Test if the data folder exists"""
    assert os.path.exists(_PATH_DATA)

@data_there
def test_train_processed_exist():
    """Test if the train_processed.pt exists"""
    assert os.path.exists(os.path.join(_PATH_DATA+"/processed/train_processed.pt"))

@data_there
def test_test_processed_exist():
    """Test if the test_processed.pt exists"""
    assert os.path.exists(os.path.join(_PATH_DATA+ "/processed/test_processed.pt"))

@data_there
## test that both the images and labels are of type torch.Tensor
def test_train_data_torch_size():
    assert train_data[0][0].shape == torch.Size([1, 28, 28])

@data_there
def test_test_data_torch_size():
    assert test_data[0][0].shape == torch.Size([1, 28, 28])

@data_there
## test that the shape of the train and test images are correct
def test_train_data_shape():
    assert len(train_data) == 25000
    assert len(test_data) == 5000

@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 54)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected




