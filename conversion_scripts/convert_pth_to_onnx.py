import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch
import torch.onnx
from src.models.model import FashionClassifier, AttentionBlock
import onnx
import onnxruntime
import numpy as np

def convert_pth_to_onnx(pth_path, onnx_path):
    # Load checkpoint
    print(f"Loading checkpoint from: {pth_path}")
    checkpoint = torch.load(pth_path, map_location='cpu')
    num_classes = len(checkpoint['classes'])
    
    # Initialize model
    print("Initializing model...")
    model = FashionClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    print("Creating dummy input...")
    x = torch.randn(1, 3, 177, 177)  # Match your training input size
    
    # Export to ONNX
    print("Converting to ONNX...")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        x,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Test with ONNX Runtime
    print("Testing with ONNX Runtime...")
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    
    print("ONNX model exported and verified successfully!")
    return checkpoint['classes']

if __name__ == "__main__":
    pth_path = os.path.join(project_root, "notebooks", "fashion_classifier_model", 
                           "run_20241130_033009", "best_model.pth")
    onnx_path = os.path.join(project_root, "converted_models", "model-1.onnx")
    
    print(f"PTH path exists: {os.path.exists(pth_path)}")
    
    try:
        classes = convert_pth_to_onnx(pth_path, onnx_path)
        print("\nConversion completed!")
        print(f"Classes: {classes}")
        
        # Save class names
        classes_path = os.path.join(project_root, "converted_models", "classes-1.txt")
        with open(classes_path, 'w') as f:
            f.write('\n'.join(classes))
            
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        raise e