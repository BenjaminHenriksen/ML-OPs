from tests import _PATH_DATA
import os
import torch
from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel
import pytest


## import the train and test data
train_data = mnist(train=True, in_folder=_PATH_DATA + "/raw", out_folder=_PATH_DATA + "/processed")
test_data = mnist(train=False, in_folder=_PATH_DATA + "/raw", out_folder=_PATH_DATA + "/processed")

## iport model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MyAwesomeModel()
model = model.to(device)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def test_model_input_output():
    for batch in dataloader:
        optimizer.zero_grad()
        x, y = batch
        preds = model(x.to(device))
        assert len(x) == len(preds)

def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    
def test_error_on_wrong_shape2():   
    with pytest.raises(ValueError, match='Expected each sample to have shape 1, 28, 28'):
        model(torch.randn(1,2,4,6))





        
        
