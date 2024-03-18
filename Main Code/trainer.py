# This is the file in which we will define the training functionality

# Imports
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# I'm not entirely sure this will work, just basing it off the torch documentation
best_model_path = "model_checkpoints/best_model.pt"

# Just set some dummy values for the moment
def fit(model, X, y, device, time_bin=None, epochs=10, batch_size=64, lr=1e-3, weight_decay=0, valid_size=0.2, print_acc=False):
    """
    Trains the model on the provided data and returns the best validation accuracy

    Args:
        model (torch.nn.Module)
        X (numpy.ndarray): Input data 
        y (numpy.ndarray): Target labels 
        device (torch.device): The device (CPU or GPU) on which to perform training.
        epochs (int) optional w default 10
        batch_size (int) optional w default 64
        lr (float) optional w default 1e-3
        weight_decay (float) optional w default 0
        valid_size (float) optional w default 20% of data set
        random_state (int) optional w default 0
        print_acc (bool) optional w default false, just adds more printing

    Returns:
        best accuracy as float
    """

    # Convert data to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Combine X and y into a TensorDataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Define the sizes of the splits
    train_size = int((1 - valid_size) * len(dataset))
    val_size = len(dataset) - train_size

    # Use random_split to create random train and validation subsets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Prepare dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Perform model training
    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0

    train_accs=[]
    valid_accs=[]

    if time_bin is None:
        time_bin = len(train_dataloader)

    for _ in tqdm(range(epochs)):
        train_acc = train(model, train_dataloader, optimizer, time_bin, device)
        valid_acc, _ = evaluate(model, val_dataloader, device)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        if print_acc:
            print(f"{train_acc=}, {valid_acc=}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), best_model_path)

    # load the best model back
    model.load_state_dict(torch.load(best_model_path))
    print(f"Best valid accuracy: {best_acc}")
    
    return best_acc, train_accs, valid_accs

def train(model, dataloader, optimizer, time_bin, device):
    """
    Training function that trains over a single epoch

    Args
        model (torch.nn.Module)
        dataloader (torch.utils.data.DataLoader) gives us the batches of training data.
        optimizer (torch.optim.Optimizer)
        device (torch.device): The device (CPU or GPU) on which to perform training.

    Returns
        Accuracy float
    """

    # Inits
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    correct = total = 0

    batches_trained = 0
    for batch_X, batch_y in dataloader:
        batches_trained += 1
        # Fresh gradients
        optimizer.zero_grad()

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_X) # forward pass
        
        # Get & handle predictions
        _, batch_y_pred = torch.max(output, 1)
        correct += (batch_y_pred == batch_y).sum().item()
        total += batch_y.size(0)

        # Loss calc, backward pass, update parameters
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        if batches_trained > time_bin:
            break

    accuracy = correct / total 
    return round(accuracy, 5)

def evaluate(model, dataloader, device):
    """
    Evaluate the model performance

    Args:
        model (torch.nn.Module)
        dataloader (torch.utils.data.DataLoader) gives us the batches of val data.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
        batch_size (int)

    Returns:
        Accuracy + predicted lables tuple
    """

    # Inits
    model.eval() 
    correct = total = 0 
    y_pred = []

    for batch_X, batch_y in dataloader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Disable gradient computation, then perform forward & get pred
        with torch.no_grad(): 
            output = model(batch_X)
            _, batch_y_pred = torch.max(output, 1)

        # Update and calc predictions
        correct += (batch_y_pred == batch_y).sum().item() 
        total += batch_y.size(0) 
        y_pred += batch_y_pred.tolist()

    # calc accuracy
    accuracy = correct / total 
    return round(accuracy, 5), y_pred 