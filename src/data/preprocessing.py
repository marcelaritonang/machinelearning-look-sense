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

        print(f"Loaded {len(self.images)} images total")

    def apply_enhanced_augmentation(self, img, class_name):
        """Enhanced augmentation based on class"""
        height, width = img.shape[:2]
        
        if class_name == 'Bottomwear':
            # Specific augmentation for Bottomwear
            angle = random.uniform(-20, 20)
            scale = random.uniform(0.8, 1.2)
            center = (width/2, height/2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            img = cv2.warpAffine(img, M, (width, height))
            
            # Add vertical stretch/compress
            new_height = int(height * random.uniform(0.9, 1.1))
            img = cv2.resize(img, (width, new_height))
            img = cv2.resize(img, (width, height))
            
        elif class_name == 'Headwear':
            # Specific augmentation for Headwear
            angle = random.uniform(-30, 30)
            scale = random.uniform(0.9, 1.1)
            center = (width/2, height/2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            img = cv2.warpAffine(img, M, (width, height))
            
            # Random zoom
            zoom = random.uniform(0.8, 1.0)
            new_width = int(width * zoom)
            new_height = int(height * zoom)
            x = (width - new_width) // 2
            y = (height - new_height) // 2
            img = img[y:y+new_height, x:x+new_width]
            img = cv2.resize(img, (width, height))
        
        
        # Brightness and contrast
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-10, 10)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Color jittering
        if random.random() > 0.5:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:,:,1] = img[:,:,1] * random.uniform(0.8, 1.2)
            img[:,:,2] = img[:,:,2] * random.uniform(0.8, 1.2)  
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            
        return img
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
            class_name = self.classes[label]
            if class_name in ['Bottomwear', 'Headwear']:
                if random.random() > 0.3:  # 70% 
                    img = self.apply_enhanced_augmentation(img, class_name)
            elif random.random() > 0.5:  # 50%
                img = self.apply_enhanced_augmentation(img, class_name)
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return np.zeros((224, 224, 3), dtype=np.uint8), label
    
    def get_class_counts(self):
        """Return dictionary of class counts"""
        return dict(Counter([self.classes[label] for label in self.labels]))