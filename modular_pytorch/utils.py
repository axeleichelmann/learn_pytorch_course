"""
This script contains various utility functions for PyTorch model training & saving
"""
import torch
import torch.nn as nn
from pathlib import Path

def save_model(model : nn.Module,
               save_dir : str,
               model_name : str):
  """
  Saves a PyTorch model to a target directory
  --------------------------------------------------------
  Inputs:
  model - the model to be saved
  save_dir - the directory under which we want to save the model
  model_name - the name we want to assign to the file containing the saved model (should include '.pth' or '.pt' file extension name)
  """

  # Create saving directory
  save_dir_path = Path(save_dir)
  save_dir_path.mkdir(parents=True, exist_ok=True)

  # Create model save path
  assert model_name.endswith('.pth') or model_name.endswith('.pt'), "model_name should end with '.pth' or '.pt' file extension"
  model_save_path = save_dir_path / model_name

  # Save the model's state_dict()
  print(f"[INFO] Saving model to {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)
