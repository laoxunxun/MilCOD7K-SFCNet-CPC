# -*- coding: utf-8 -*-
"""
Multi-class segmentation data loader
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import cv2


class SalObjDatasetMultiClass(data.Dataset):
    def __init__(self, image_root, gt_root, edge_root, trainsize, num_classes=4, augmentation=True):
        self.trainsize = trainsize
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.endswith(('.jpg', '.png'))])
        self.gts = sorted([os.path.join(gt_root, f.replace('.jpg', '.npy').replace('.png', '.npy'))
                           for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))])
        self.edges = sorted([os.path.join(edge_root, f.replace('.jpg', '.png').replace('.png', '.png'))
                             for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))])
        self.filter_files()
        self.size = len(self.images)

        # base transform (Resize + Normalize) for the augmentation-free validation set
        self.base_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.edges_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()
        ])
        print(f"Dataset loaded: {self.size} images, {self.num_classes} classes, augmentation={self.augmentation}")

    def filter_files(self):
        images, gts, edges = [], [], []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            if not all(os.path.exists(p) for p in [img_path, gt_path, edge_path]):
                continue
            try:
                gt_data = np.load(gt_path)
                if gt_data.shape[2] != self.num_classes:
                    continue
                Image.open(img_path).load()
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
            except Exception:
                continue
        self.images, self.gts, self.edges = images, gts, edges

    def _augment(self, image, gt_np, edge):
        """Apply synchronized augmentation to image (PIL), gt_np (H,W,C ndarray), and edge (PIL)"""
        # 1. random horizontal flip (50%)
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            gt_np = gt_np[:, ::-1, :].copy()
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. random vertical flip (30%)
        if random.random() > 0.7:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            gt_np = gt_np[::-1, :, :].copy()
            edge = edge.transpose(Image.FLIP_TOP_BOTTOM)

        # 3. random rotation (90/180/270 or small angles)
        k = random.choice([0, 0, 0, 1, 2, 3])  # 0 has a higher probability
        if k > 0:
            image = image.rotate(k * 90, expand=True)
            gt_np = np.rot90(gt_np, k, axes=(0, 1)).copy()
            edge = edge.rotate(k * 90, expand=True)

        # 4. color jitter (brightness, contrast, saturation)
        if random.random() > 0.3:
            color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
            image = color_jitter(image)

        # 5. Gaussian blur (10%)
        if random.random() > 0.9:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        return image, gt_np, edge

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt_data = np.load(self.gts[index])  # H x W x C, uint8, 0/1
        edge = self.binary_loader(self.edges[index])

        # apply data augmentation during training
        if self.augmentation:
            image, gt_data, edge = self._augment(image, gt_data, edge)

        # Transform image
        image = self.base_transform(image)

        # Transform GT using cv2 (handles multi-channel arrays properly)
        gt_resized = cv2.resize(gt_data, (self.trainsize, self.trainsize), interpolation=cv2.INTER_NEAREST)
        gt_tensor = torch.from_numpy(gt_resized).float()  # H x W x C, float 0-1 (already 0/1 values)
        gt_tensor = gt_tensor.permute(2, 0, 1)  # C x H x W

        # edge is already PIL Image from binary_loader
        edge_tensor = self.edges_transform(edge)
        return image, gt_tensor, edge_tensor

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


def get_loader_multiclass(image_root, gt_root, edge_root, batchsize, trainsize, num_classes=4,
                          shuffle=True, num_workers=0, pin_memory=True, augmentation=True):
    dataset = SalObjDatasetMultiClass(image_root, gt_root, edge_root, trainsize, num_classes,
                                      augmentation=augmentation)
    return data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)


class test_dataset_multiclass:
    def __init__(self, image_root, testsize, num_classes=4):
        self.testsize = testsize
        self.num_classes = num_classes
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.size = len(self.images)
        self.index = 0
        print(f"Test dataset: {self.size} images")

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        self.index = (self.index + 1) % self.size
        return self.transform(image).unsqueeze(0), os.path.basename(self.images[self.index - 1])

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def __len__(self):
        return self.size
