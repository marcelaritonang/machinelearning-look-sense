import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import random
from collections import Counter

class FashionDataset(Dataset):
    def __init__(self, data_dir, transform=None, samples_per_class=200):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples_per_class = samples_per_class
        
        # Get class names (folders)
        self.classes = [d.name for d in self.data_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        self.classes.sort()  # Ensure consistent order
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        print("Loading dataset...")
        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            # Support multiple image formats
            img_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                img_paths.extend(list(cls_dir.glob(ext)))
            
            if len(img_paths) > samples_per_class:
                print(f"Limiting {cls_name} to {samples_per_class} samples from {len(img_paths)}")
                img_paths = random.sample(img_paths, samples_per_class)
            else:
                print(f"Using all {len(img_paths)} samples for {cls_name}")
            
            for img_path in img_paths:
                self.images.append(str(img_path))
                self.labels.append(self.class_to_idx[cls_name])
        
        # Verify images
        self._verify_images()
        print(f"Loaded {len(self.images)} images total")
    
    def _verify_images(self):
        """Verify images are readable"""
        valid_images = []
        valid_labels = []
        for img_path, label in zip(self.images, self.labels):
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    valid_images.append(img_path)
                    valid_labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
        
        self.images = valid_images
        self.labels = valid_labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return a default image in case of error
            return np.zeros((128, 128, 3), dtype=np.uint8), label
    
    def get_class_counts(self):
        """Return dictionary of class counts"""
        return dict(Counter([self.classes[label] for label in self.labels]))