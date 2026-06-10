"""
train_resnet.py
===============
Trains ResNet-50 to predict wind streak direction (0-180°)
from Sentinel-1 SAR patches, with data augmentation.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn.functional as F

class SARWindDataset(Dataset):
    def __init__(self, patch_dir, labels_path, augment=False):
        self.patches = sorted([
            os.path.join(patch_dir, f)
            for f in os.listdir(patch_dir) if f.endswith('.npy')
        ])
        self.labels = np.load(labels_path).astype(np.float32)
        self.augment = augment

        # Match patches to labels (in case some failed)
        min_len = min(len(self.patches), len(self.labels))
        self.patches = self.patches[:min_len]
        self.labels  = self.labels[:min_len]

    def __len__(self):
        # Return 8x the real size due to augmentation
        return len(self.patches) * (8 if self.augment else 1)

    def __getitem__(self, idx):
        real_idx = idx % len(self.patches)
        aug_idx  = idx // len(self.patches)  # 0-7 augmentation variant

        arr = np.load(self.patches[real_idx]).astype(np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)

        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
        t = t.squeeze(0)  # 1,224,224

        # Apply 8 deterministic augmentations
        label_offset = 0.0
        if aug_idx == 1:
            t = torch.flip(t, [2])           # horizontal flip
            label_offset = 0.0
        elif aug_idx == 2:
            t = torch.flip(t, [1])           # vertical flip
            label_offset = 0.0
        elif aug_idx == 3:
            t = torch.rot90(t, 1, [1, 2])    # rotate 90°
            label_offset = 90.0
        elif aug_idx == 4:
            t = torch.rot90(t, 2, [1, 2])    # rotate 180°
            label_offset = 0.0
        elif aug_idx == 5:
            t = torch.rot90(t, 3, [1, 2])    # rotate 270°
            label_offset = 90.0
        elif aug_idx == 6:
            t = torch.flip(torch.rot90(t, 1, [1, 2]), [2])
            label_offset = 90.0
        elif aug_idx == 7:
            t = torch.flip(torch.rot90(t, 1, [1, 2]), [1])
            label_offset = 90.0

        t = t.repeat(3, 1, 1).unsqueeze(0)  # 1,3,224,224
        t = t.squeeze(0)

        label = torch.tensor((self.labels[real_idx] + label_offset) % 180)
        return t, label


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    dataset = SARWindDataset('data/patches', 'data/labels.npy', augment=True)
    print(f"Dataset size after augmentation: {len(dataset)} samples")
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(30):  # more epochs
        model.train()
        total_loss = 0
        for patches, labels in loader:
            patches, labels = patches.to(device), labels.to(device)
            pred = model(patches).squeeze(1)
            loss = loss_fn(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_resnet50_direction_model.pth')

    print(f"Training complete. Best loss: {best_loss:.4f}")
    print("Saved best model.")

if __name__ == '__main__':
    train()