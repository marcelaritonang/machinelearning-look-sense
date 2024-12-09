import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.data.preprocessing import FashionDataset
from src.models.model import FashionClassifier
import os
from tqdm import tqdm

def test_model(model_path, test_data_dir):
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")

        # Load the saved model
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get classes from checkpoint
        classes = checkpoint['classes']
        num_classes = len(classes)
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {classes}")
        
        # Initialize model
        model = FashionClassifier(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # Define transforms
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((177, 177)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load test dataset
        print(f"\nLoading test data from: {test_data_dir}")
        test_dataset = FashionDataset(test_data_dir, transform=test_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )

        # Initialize metrics
        correct = 0
        total = 0
        class_correct = {class_name: 0 for class_name in classes}
        class_total = {class_name: 0 for class_name in classes}

        # Testing loop
        print("\nStarting testing...")
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing progress"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i]
                    pred = predicted[i]
                    class_name = classes[label]
                    class_total[class_name] += 1
                    if label == pred:
                        class_correct[class_name] += 1

        # Calculate and print overall accuracy
        overall_accuracy = 100 * correct / total
        print(f'\nOverall Accuracy: {overall_accuracy:.2f}%')

        # Print per-class accuracy
        print("\nPer-class Accuracy:")
        for class_name in classes:
            accuracy = 100 * class_correct[class_name] / class_total[class_name]
            print(f"{class_name}: {accuracy:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")

        return overall_accuracy

    except Exception as e:
        print(f"\nError occurred during testing: {str(e)}")
        raise e

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Fashion Classifier Model')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test data directory')
    
    args = parser.parse_args()
    
    try:
        test_model(args.model_path, args.test_data)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise e