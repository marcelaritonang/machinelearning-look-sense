import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.preprocessing import FashionDataset
from src.models.model import FashionClassifier
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Perbaikan import

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load dataset with limited samples per class
    dataset = FashionDataset('Dataset', transform=transform, samples_per_class=800)
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    # Print samples per class
    class_counts = dataset.get_class_counts()
    print("\nSamples per class:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders with GPU optimizations
    train_loader = DataLoader(train_dataset, 
                            batch_size=64,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)
    
    val_loader = DataLoader(val_dataset,
                          batch_size=64,
                          num_workers=4,
                          pin_memory=True)

    # Initialize model on GPU
    num_classes = len(dataset.classes)
    model = FashionClassifier(num_classes=num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)  # Perbaikan penggunaan

    # Training loop
    num_epochs = 20
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        accuracy = 100 * correct / total

        # Update learning rate
        scheduler.step(val_loss)

        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Ensure directory exists
            os.makedirs('notebooks/fashion_classifier_model', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'classes': dataset.classes
            }, 'notebooks/fashion_classifier_model/best_model.pth')
            print(f"Saved best model with validation loss: {val_loss:.4f}")

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('notebooks/fashion_classifier_model/training_history.png')
    plt.show()

if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise e