device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_classes = 3

model = UNetMultiClass(n_channels=3, n_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
criterion = nn.CrossEntropyLoss(ignore_index=255)
import torch

def dice_score_multiclass(preds, targets, n_classes=3):
    """
    preds: [B,H,W] после argmax
    targets: [B,H,W]
    """
    preds = preds.cpu()
    targets = targets.cpu()
    dice = 0.0
    for c in range(n_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice += (2 * intersection + 1e-7) / (union + 1e-7)
    return dice / n_classes

def train_one_epoch(model, loader, optimizer, device, criterion, n_classes=3):
    model.train()
    total_loss = 0
    total_dice = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)               # [B, n_classes, H, W]
        loss = criterion(outputs, masks)     # masks: [B,H,W]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        total_dice += dice_score_multiclass(preds, masks, n_classes).item()

    return total_loss / len(loader), total_dice / len(loader)


def validate(model, loader, device, criterion, n_classes=3):
    model.eval()
    total_loss = 0
    total_dice = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_dice += dice_score_multiclass(preds, masks, n_classes).item()

    return total_loss / len(loader), total_dice / len(loader)
