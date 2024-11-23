import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from pathlib import Path
import json
import yaml

def load_config(config_path: str) -> Dict:
    """
    Load configuration file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
    raise ValueError("Unsupported config file format")

def save_metrics(metrics: Dict, save_path: str):
    """
    Save training metrics
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save metrics
    """
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history
    
    Args:
        history: Dictionary of metrics lists
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))
    for metric_name, metric_values in history.items():
        plt.plot(metric_values, label=metric_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training History')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        labels: Array of labels
        
    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels)
    total = len(labels)
    class_weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)

def create_checkpoint(model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     epoch: int,
                     loss: float,
                     save_path: str):
    """
    Create model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   checkpoint_path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint
        
    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

# Example usage
if __name__ == "__main__":
    # Load config
    config = load_config('config.yaml')
    
    # Example training history plot
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.55, 0.45, 0.35]
    }
    plot_training_history(history, 'training_history.png')