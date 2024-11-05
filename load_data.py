import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from coco_dataset import COCOPanopticDataset

data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize images to 512x512
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize based on ImageNet means and std
])

from torch.utils.data import DataLoader

# Paths to COCO data
train_image_dir = "datasets/coco/train2017"
train_instance_file = "datasets/coco/annotations/instances_train2017.json"
train_panoptic_file = "datasets/coco/annotations/panoptic_train2017.json"
train_panoptic_mask_dir = "datasets/coco/panoptic_train2017"

# Initialize dataset
train_dataset = COCOPanopticDataset(
    image_dir=train_image_dir,
    instance_file=train_instance_file,
    panoptic_file=train_panoptic_file,
    panoptic_mask_dir=train_panoptic_mask_dir,
    transform=data_transform
)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)