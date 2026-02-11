import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np

class OxfordPetsMultiClassDataset(Dataset):
    """
    Мультиклассовый датасет Oxford Pets.
    Маски: 1=фон, 2=кошки, 3=собаки → 0,1,2
    """

    def __init__(self, root, joint_transform=None, image_transform=None):
        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "annotations", "trimaps")

        images = [f for f in os.listdir(self.image_dir) if f.lower().endswith(".jpg")]
        masks = {os.path.splitext(f)[0]: f for f in os.listdir(self.mask_dir)}

        self.images = []
        self.masks = []
        for img in images:
            name = os.path.splitext(img)[0]
            if name in masks:
                self.images.append(img)
                self.masks.append(masks[name])

        self.images.sort()
        self.masks.sort()

        self.joint_transform = joint_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.image_dir, self.images[idx])
        ).convert("RGB")

        mask = Image.open(
            os.path.join(self.mask_dir, self.masks[idx])
        )

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.image_transform:
            image = self.image_transform(image)

        mask = torch.from_numpy(np.array(mask)).long()

        # original: 1=bg, 2=pet, 3=border
        mask = mask - 1  # -> 0,1,2

        # border -> ignore
        mask[mask == 2] = 255

        mask[mask < 0] = 255
        mask[mask > 1] = 255

        return image, mask