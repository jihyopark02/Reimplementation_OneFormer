import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class COCOPanopticDataset(Dataset):
    def __init__(self, image_dir, instance_file, panoptic_file, panoptic_mask_dir, transform=None, max_samples=5000):
        """
        Args:
            image_dir (str): Directory with images.
            instance_file (str): Path to the JSON file with instance annotations.
            panoptic_file (str): Path to the JSON file with panoptic annotations.
            panoptic_mask_dir (str): Directory with panoptic segmentation masks.
            transform (callable, optional): Optional transform to be applied on an image.
            max_samples (int, optional): Maximum number of samples to load. Defaults to 5000.
        """
        self.image_dir = image_dir
        self.instance_file = instance_file
        self.panoptic_file = panoptic_file
        self.panoptic_mask_dir = panoptic_mask_dir
        self.transform = transform
        self.max_samples = max_samples

        # Load instance and panoptic annotations
        with open(self.instance_file, 'r') as f:
            self.instance_annotations = json.load(f)

        with open(self.panoptic_file, 'r') as f:
            self.panoptic_annotations = json.load(f)

        # List of image information, limited by max_samples
        self.images = self.instance_annotations['images'][:self.max_samples]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_filename = img_info['file_name']

        # Load image
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")

        # Load panoptic segmentation mask
        panoptic_mask_path = os.path.join(self.panoptic_mask_dir, f"{img_id}.png")
        try:
            panoptic_mask = Image.open(panoptic_mask_path)
        except FileNotFoundError:
            print(f"Panoptic mask {img_id}.png not found in {self.panoptic_mask_dir}")
            panoptic_mask = Image.new("L", (512, 512))  # Create a blank mask as a placeholder

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            panoptic_mask = torch.tensor(np.array(panoptic_mask), dtype=torch.long)

        return image, panoptic_mask
