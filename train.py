import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.preprocessing import FashionDataset
from src.models.model import FashionClassifier
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', marker='o', linestyle='-', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', marker='s', linestyle='-', linewidth=2)
    plt.title('Model Loss', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', marker='o', linestyle='-', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', marker='s', linestyle='-', linewidth=2)
    plt.title('Model Accuracy', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_model():
    # Set device and seeds
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # Create save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('notebooks', 'fashion_classifier_model', f'run_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = FashionDataset('Dataset', transform=train_transform, samples_per_class=200)
    val_dataset = FashionDataset('Dataset', transform=val_transform, samples_per_class=200)

    print(f"\nTotal samples: {len(train_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    class_counts = train_dataset.get_class_counts()
    print("\nSamples per class:")
    for cls, count in class_counts.items():
        print(f"{cls}: {count}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True
    )

    # Initialize model
    model = FashionClassifier(num_classes=len(train_dataset.classes)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )

    # Training parameters
    num_epochs = 30
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    print("\nStarting training...")
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^8} | {'Train Acc':^9} | {'Val Acc':^8}")
    print("-" * 55)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct_train / total_train:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        class_correct = [0] * len(train_dataset.classes)
        class_total = [0] * len(train_dataset.classes)
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Update learning rate
        scheduler.step(val_loss)

        # Print results
        print(f"{epoch+1:6d} | {train_loss:10.4f} | {val_loss:8.4f} | "
              f"{train_accuracy:8.2f}% | {val_accuracy:7.2f}%")

        print("\nPer-class Validation Accuracy:")
        for i in range(len(train_dataset.classes)):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'{train_dataset.classes[i]}: {class_acc:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            counter = 0
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'classes': train_dataset.classes
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"\nSaved best model with validation loss: {val_loss:.4f} "
                  f"and accuracy: {val_accuracy:.2f}%")
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break

        print("-" * 55)

    # Plot final metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir)

    # Save final metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss
    }
    torch.save(metrics, os.path.join(save_dir, 'training_metrics.pth'))

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation accuracy: {val_accuracy:.2f}%")

if __name__ == '__main__':
    try:
        train_model()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise e