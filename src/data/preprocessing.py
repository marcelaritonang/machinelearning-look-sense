import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import random
from collections import Counter

class FashionDataset(Dataset):
    def __init__(self, data_dir, transform=None, samples_per_class=800):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples_per_class = samples_per_class
        
        # Get class names (folders)
        self.classes = [d.name for d in self.data_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        self.classes.sort()  # Sort for consistency
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load and limit data paths
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            img_paths = list(cls_dir.glob('*.jpg'))  # Adjust pattern if needed
            
            # Randomly sample if more than limit
            if len(img_paths) > samples_per_class:
                img_paths = random.sample(img_paths, samples_per_class)
            
            for img_path in img_paths:
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[cls_name])
        
        # Shuffle the dataset
        combined = list(zip(self.images, self.labels))
        random.shuffle(combined)
        self.images, self.labels = zip(*combined)
        
        print(f"Loaded {len(self.images)} images total")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_class_counts(self):
        """Return dictionary of class counts"""
        return dict(Counter([self.classes[label] for label in self.labels]))