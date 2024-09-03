"""
This file contains the functionality for creating PyTorch DataLoaders for
image classification data.
"""


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir : str, test_dir : str,
                       train_transform : transforms.Compose, test_transform : transforms.Compose,
                       batch_size : int, num_workers : int):
  """
  This function creates dataloaders for training and evaluating a pytorch model

  ------------------------------------------------------------------------------
  Inputs:

  train_dir - The path of the diredctory containing the training image data
  test_dir - The path of the directory containing the testing image data
  train_transform - The transformation we want to apply to the training image data
  test_transform - The transformation we want to apply to the testing image data
  batch_size - The size of the mini-batches in the dataloaders
  num_workers - The number of workers assigned to create the dataloaders

  ------------------------------------------------------------------------------
  Outputs:

  train_dataloader - the training dataloader (with shuffling applied)
  test_dataloader - the testing dataloader
  class_names - the names of the different classes in the training & testing data
  """

  # Create training & testing datasets using `ImageFolder` function
  train_data = datasets.ImageFolder(train_dir,
                                    transform=train_transform,
                                    target_transform=None)

  test_data = datasets.ImageFolder(test_dir,
                                   transform=test_transform)

  # Get class names
  class_names = train_data.classes

  # Create training & testing dataloaders
  train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)

  test_dataloader = DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers = num_workers)

  return train_dataloader, test_dataloader, class_names
