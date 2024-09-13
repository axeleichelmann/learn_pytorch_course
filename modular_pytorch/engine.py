"""
Contains functions for model training and evaluation within each epoch, and a function for the overall model training across multiple epochs
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm


def train_step(model : nn.Module,
               train_dataloader : DataLoader,
               device : torch.device,
               optimizer : torch.optim,
               loss_fn):
  """
  Carries out the training step on a pytorch model & calculates training loss & accuracy

  --------------------------------
  Inputs:
  model - the model to be trained
  train_dataloader - the dataloader containing the training data
  device - the device on which the model exists
  optimizer - the optimizer to use for model training
  loss_fn - the loss function used to evaluate the model's success
  acc_fn - the accuracy function used to evaluate the model's success


  --------------------------------
  Outputs: Tuple[loss, acc]
  loss - the training loss of the model
  acc - the training accuracy of the model
  """

  # Initialize loss & accuracy
  train_loss, train_acc = 0, 0

  # Set model to training mode
  model.train()
  for images, labels in train_dataloader:
    images, labels = images.to(device), labels.to(device)  # Move batch images & labels to device

    y_logits = model(images)   # Carry out forward pass

    loss = loss_fn(y_logits, labels)   # Calculate batch loss
    train_loss += loss    # Update overall epoch loss

    # Calculate and accumulate accuracy metric across all batches
    y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
    train_acc += (y_pred_class == labels).sum().item()/len(y_logits)

    # Carry out model training given batch loss
    optimizer.zero_grad()  # Zero the optimizer gradient
    loss.backward()  # Carry out backpropagation
    optimizer.step()  # Update weights


  train_loss /= len(train_dataloader)  # Calculate average loss across the dataloader images
  train_acc /= len(train_dataloader)   # Calculate average accuracy across the dataloader images

  return train_loss, train_acc



def eval_step(model : nn.Module,
               test_dataloader : DataLoader,
               device : torch.device,
               loss_fn):
  """
  Carries out the evaluation step on a pytorch model using model.eval() mode with torch.inference_mode() by
  calculating the testing loss & accuracy obtained when predicting unseen data classes

  --------------------------------
  Inputs:
  model - the model to be trained
  test_dataloader - the dataloader containing the unseen testing data
  device - the device on which the model exists
  loss_fn - the loss function used to evaluate the model
  acc_fn - the accuracy function used to evaluate the model


  --------------------------------
  Outputs: Tuple[loss, acc]
  test_loss - the loss of the model when prediciting the unseen data classes
  test_acc - the accuracy of the model in predicting the unseen data classes
  """

  # Initialize loss & accuracy
  test_loss, test_acc = 0, 0

  # Set model to evaluation mode
  model.eval()
  with torch.inference_mode():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)   # Move batch images & labels to device

        y_logits = model(images)   # Carry out forward pass

        loss = loss_fn(y_logits, labels)   # Calculate batch loss
        test_loss += loss.item()    # Update overall epoch loss


        test_pred_labels = y_logits.argmax(dim=1)
        test_acc += ((test_pred_labels == labels).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(test_dataloader)  # Calculate average loss across the dataloader images
    test_acc = test_acc / len(test_dataloader)   # Calculate average accuracy across the dataloader images

  return test_loss, test_acc



def train_model(model : nn.Module, num_epochs : int,
                train_dataloader : DataLoader, test_dataloader : DataLoader,
                optimizer : torch.optim,
                device : torch.device,
                loss_fn):
  """
  Trains the model for a given numebr of epochs & calculates the training and testing loss & accuracy of the model at each epoch in the training process

  ------------------------------------------------------------------------
  Inputs:
  model - model to be trained
  num_epochs - the number of epochs for which we want to train the model
  train_dataloader - dataloader contatining the training data in batched format
  test_dataloader - dataloader containing the testing data in batched format
  optimizer - the optimizer to use for model training
  device - device on which the model exists & should be trained on
  loss_fn - the loss function used to evaluate the model
  acc_fn - the accuracy function used to evaluate the model

  ------------------------------------------------------------------------
  Outputs:
  Tuple of lists in the form [train_losses, train_accs, eval_losses, eval_accs]
  Each element of the tuple is a list containing the following info:
  train_losses - contains the training loss observed at each epoch
  eval_losses - contains the testing loss observed at each epoch
  train_accs - contains the training accuracies observed at each epoch
  eval_losses - contains the testing accuracies observed at each epoch
  """


  # Create lists in which to store model training & evaluation results
  train_losses, train_accs = [], []
  eval_losses, eval_accs = [], []

  for epoch in tqdm(range(num_epochs)):
    train_results = train_step(model,
                               train_dataloader,
                               device, optimizer,
                               loss_fn)

    eval_results = eval_step(model,
                             test_dataloader,
                             device,
                             loss_fn)

    # Print results every 10 epochs
    if epoch % 10 == 0:
      print(f"Epoch : {epoch} | Train Loss = {train_results[0]:.4f}, Train Acc = {train_results[1]:2f} | Test Loss = {eval_results[0]:.4f}, Test Acc = {eval_results[1]:.2f}")

    # Append training & evaluation results to lists
    train_losses.append(train_results[0])
    train_accs.append(train_results[1])

    eval_losses.append(eval_results[0])
    eval_accs.append(eval_results[1])

  return train_losses, train_accs, eval_losses, eval_accs


# Define new `train_model` function that includes SummaryWriter
def train_model_v2(model : nn.Module, num_epochs : int,
                   train_dataloader : DataLoader, test_dataloader : DataLoader,
                   optimizer : torch.optim, device : torch.device, loss_fn,
                   writer : torch.utils.tensorboard):
  """
  Trains & Evaluates Model, Stores Results in designated directory, and returns lists containign training & evaluation results
  ----------------------------------------------
  Input: 
  model - model to be trained & evaluated
  num_epochs - numer of epochs for which you want to train the model
  train_dataloader - dataloader containing training data in batch format
  test_dataloader - dataloader containing testing data in batch format
  optimizer - optimizer to use for model training
  device - device on which the carry out model training
  loss_fn - loss function for evaluating model
  writer - writer used to store training & evaluation results

  ------------------------------------------------
  Output:
  Tuple[train_losses, train_accs, eval_losses, eval_accs]
  """
  
  # Create lists in which to store training & validation losses and accuracies
  train_losses, train_accs = [], []
  eval_losses, eval_accs = [], []

  for epoch in tqdm(range(num_epochs)):
    train_loss, train_acc = train_step(model, train_dataloader, device, optimizer, loss_fn)
    eval_loss, eval_acc = eval_step(model, test_dataloader, device, loss_fn)

    # Append training & evaluation results to lists
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    eval_losses.append(eval_loss)
    eval_accs.append(eval_acc)

    # Print results every 10 epochs
    if epoch % 10 == 0:
      print(f"Epoch : {epoch} | Training Loss = {train_loss:.3f}, Training Accuracy = {train_acc:.2f} | Evaluation Loss = {eval_loss:.3f}, Evaluation Accuracy = {eval_acc:.2f}")

    # Add results to SummaryWriter
    writer.add_scalars(main_tag="Loss Values", tag_scalar_dict = {"train_loss" : train_loss, "eval_loss" : eval_loss}, global_step = epoch)
    writer.add_scalars(main_tag="Accuracy Scores", tag_scalar_dict = {"train_acc" : train_acc, "eval_acc" : eval_acc}, global_step = epoch)
    writer.add_graph(model=model, input_to_model=torch.randn(8,3,224,224).to(device))

  # Close writer after all of model training
  writer.close()

  return train_losses, train_accs, eval_losses, eval_accs


                     
