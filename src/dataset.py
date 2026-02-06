# src/dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

class OxfordPetsDataset(Dataset):
    def __init__(self, root, image_transform=None, mask_transform=None):
        self.root = root
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, "annotations", "trimaps")

        # Берем только реальные файлы
        images = [f for f in os.listdir(self.image_dir) if f.lower().endswith('.jpg')]
        masks = [f for f in os.listdir(self.mask_dir) if f.lower().endswith('.png')]

        # Сопоставляем по имени без расширения
        mask_names = {os.path.splitext(f)[0]: f for f in masks}
        self.images = []
        self.masks = []
        for img in images:
            name = os.path.splitext(img)[0]
            if name in mask_names:
                self.images.append(img)
                self.masks.append(mask_names[name])

        self.images.sort()
        self.masks.sort()

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = torch.from_numpy(np.array(mask)).long()
        mask = torch.where(mask > 1, 1, 0)

        return image, mask
