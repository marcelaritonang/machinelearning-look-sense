import torch
import gc

def clear_gpu_memory():
    if torch.cuda.is_available():
        # Clear GPU cache
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        print("GPU memory cleared!")
    else:
        print("No GPU available")

if __name__ == "__main__":
    clear_gpu_memory()