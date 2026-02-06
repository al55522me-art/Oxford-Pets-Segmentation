# src/train_utils.py
import torch
from .losses import segmentation_loss

def dice_score(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()
    intersection = (preds*targets).sum(dim=(2,3))
    union = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    dice = (2*intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = segmentation_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.unsqueeze(1).to(device)
            outputs = model(images)
            val_loss += segmentation_loss(outputs, masks).item()
            val_dice += dice_score(outputs, masks)
    return val_loss/len(loader), val_dice/len(loader)
