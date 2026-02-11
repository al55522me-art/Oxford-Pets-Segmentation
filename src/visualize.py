import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_prediction(model, dataset, device, idx=None):
    model.eval()

    if idx is None:
        idx = np.random.randint(0, len(dataset))

    image, mask = dataset[idx]

    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_input)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu()

    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    pred_np = pred.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="jet")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_np, cmap="jet")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.savefig("prediction3.png")
    plt.show()


visualize_prediction(model, val_dataset, device)

def visualize_with_overlay(model, dataset, device, idx=None):
    model.eval()

    if idx is None:
        idx = np.random.randint(0, len(dataset))

    image, mask = dataset[idx]
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_input)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu()

    image_np = image.permute(1, 2, 0).cpu().numpy()
    pred_np = pred.numpy()

    plt.figure(figsize=(6,6))
    plt.imshow(image_np)
    plt.imshow(pred_np, alpha=0.5, cmap="jet")
    plt.axis("off")
    plt.savefig(f"pred_overlay.png", bbox_inches=None, pad_inches=0)
    plt.show()

#%%
visualize_with_overlay(model, val_dataset, device)
