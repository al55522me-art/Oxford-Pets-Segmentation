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