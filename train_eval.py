import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score


def train_step(model: nn.Module,
               loader: data.DataLoader,
               optimizer: torch.optim,
               max_grad_norm: int,
               criterion: nn.Module,
               device: str):
    """
    Train the model on the given data for one epoch.

    Args:
        model (nn.Module): the neural network model to be trained
        loader (data.DataLoader): the data loader object that provides the training data
        optimizer (torch.optim): the optimizer object that updates the model parameters
        max_grad_norm (int): the maximum gradient norm for gradient clipping
        criterion (nn.Module): the loss function object that computes the training loss
        device (str): the device where the computation is performed (CPU or GPU)

    Returns:
        The average loss for the epoch.
    """
    model.train()  # set the model in training mode
    tot_loss = torch.tensor(0., dtype=torch.float32)  # initialize total loss
    for i, (batch_oh_sequences, batch_oh_labels) in tqdm(enumerate(loader)):
        # move the batch data to the device
        batch_oh_sequences = batch_oh_sequences.permute(0, 2, 1).to(device=device,
                                                                    non_blocking=True)
        batch_oh_labels = torch.squeeze(batch_oh_labels, dim=1).to(device=device,
                                                                   non_blocking=True)

        # forward pass
        preds = model(batch_oh_sequences)
        loss = criterion(preds, batch_oh_labels)  # compute the loss

        # backward pass
        # as recommended by PyTorch, set_to_none=True for better performance
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # clip gradients to avoid explosion
        optimizer.step()  # update model parameters

        # update the total loss
        tot_loss += loss.item()

    # empty the cache to free up GPU memory
    torch.cuda.empty_cache()

    return tot_loss / len(loader)


def test_step(model: nn.Module,
              loader: data.DataLoader,
              criterion: nn.Module,
              device: str):
    """
    Evaluate the model on the given data for one epoch.

    Args:
        model (nn.Module): the neural network model to be evaluated
        loader (data.DataLoader): the data loader object that provides the test data
        criterion (nn.Module): the loss function object that computes the test loss
        device (str): the device where the computation is performed (CPU or GPU)

    Returns:
        The average loss and the average accuracy for the epoch.
    """

    model.eval()  # set the model in evaluation mode
    tot_loss = torch.tensor(0., dtype=torch.float32)  # initialize total loss
    accuracies = []  # initialize list of accuracies
    with torch.no_grad():
        for i, (batch_oh_sequences, batch_oh_labels) in enumerate(loader):
            # move the batch data to the device
            batch_oh_sequences = batch_oh_sequences.permute(0, 2, 1).to(device=device,
                                                                        non_blocking=True)
            batch_oh_labels = torch.squeeze(batch_oh_labels, dim=1).to(device=device,
                                                                       non_blocking=True)
            # forward pass
            preds = model(batch_oh_sequences)
            loss = criterion(preds, batch_oh_labels)  # compute the loss

            # compute accuracy
            preds = [np.where(pred == np.max(pred), 1., 0.) for pred in preds.cpu().detach().numpy()]
            accuracies.append(accuracy_score(batch_oh_labels.cpu().numpy(), preds))

            # update the total loss
            tot_loss += loss.item()

    # empty the cache to free up GPU memory
    torch.cuda.empty_cache()

    return tot_loss / len(loader), sum(accuracies) / len(accuracies)
