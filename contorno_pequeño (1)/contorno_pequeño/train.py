import torch
import segmentation_models_pytorch as smp

dice_loss = smp.losses.DiceLoss(mode='binary')
focal_loss = smp.losses.FocalLoss(mode='binary')

def loss_fn(y_pred, y_true):
    return dice_loss(y_pred, y_true) + focal_loss(y_pred, y_true)

def accuracy(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    correct = (y_pred == y_true).float().sum()
    total = y_true.numel()
    return correct / total

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for j, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        acc = accuracy(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += acc.item()

    avg_loss = total_loss / (j + 1)
    avg_accuracy = total_accuracy / (j + 1)
    return avg_loss, avg_accuracy
