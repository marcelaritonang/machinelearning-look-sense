import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class FashionClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        """
        Fashion classifier model
        
        Args:
            num_classes: Number of classification classes
        """
        super(FashionClassifier, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Class predictions
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def save_model(self, path: str, metadata: Dict = None):
        """
        Save model weights and metadata
        
        Args:
            path: Path to save model
            metadata: Additional metadata to save
        """
        save_dict = {
            'state_dict': self.state_dict(),
            'metadata': metadata
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load_model(cls, path: str, num_classes: int = 10) -> Tuple['FashionClassifier', Dict]:
        """
        Load model from file
        
        Args:
            path: Path to model file
            num_classes: Number of classes
            
        Returns:
            Model instance and metadata
        """
        save_dict = torch.load(path)
        model = cls(num_classes=num_classes)
        model.load_state_dict(save_dict['state_dict'])
        return model, save_dict.get('metadata', {})

# Training utilities
class ModelTrainer:
    def __init__(self, 
                 model: FashionClassifier,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = FashionClassifier(num_classes=10)
    
    # Example forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")