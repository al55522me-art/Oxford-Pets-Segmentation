# src/visualize.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(model, dataset, device, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]
            input_img = image.unsqueeze(0).to(device)
            output = model(input_img)
            pred = torch.sigmoid(output)[0,0].cpu().numpy()
            pred = (pred > 0.5).astype(np.uint8)
            img = image.permute(1,2,0).cpu().numpy()
            gt_mask = mask.cpu().numpy()

            ax1, ax2, ax3 = axes[i]
            ax1.imshow(img); ax1.set_title("Image"); ax1.axis("off")
            ax2.imshow(gt_mask, cmap="gray"); ax2.set_title("Ground Truth"); ax2.axis("off")
            ax3.imshow(pred, cmap="gray"); ax3.set_title("Prediction"); ax3.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_and_save_overlay(model, dataset, device, num_samples=5, output_dir="examples"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = dataset[i]
            input_img = image.unsqueeze(0).to(device)
            output = model(input_img)
            pred = torch.sigmoid(output)[0,0].cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)
            img = (image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            gt_mask = mask.cpu().numpy()

            overlay = img.copy()
            overlay[pred_bin == 1] = [255,0,0]

            fig, axes = plt.subplots(1,3, figsize=(12,4))
            axes[0].imshow(img); axes[0].set_title("Image"); axes[0].axis("off")
            axes[1].imshow(gt_mask, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
            axes[2].imshow(overlay); axes[2].set_title("Prediction Overlay"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"))
            plt.close(fig)
