import torch
import torch.nn as nn

class FashionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FashionClassifier, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: Input(3, 128, 128) -> Output(32, 64, 64)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: Input(32, 64, 64) -> Output(64, 32, 32)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: Input(64, 32, 32) -> Output(128, 16, 16)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: Input(128, 16, 16) -> Output(128, 8, 8)
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Global Average Pooling: Input(128, 8, 8) -> Output(128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            # Flatten the output
            nn.Flatten(),
            
            # First fully connected layer
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second fully connected layer
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Pass input through feature extraction layers
        x = self.features(x)
        
        # Pass through classifier layers
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        """Initialize model weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)