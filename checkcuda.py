import torch

def check_cuda():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print("GPU Device:", torch.cuda.get_device_name())
        print("GPU Count:", torch.cuda.device_count())
        
        # Test tensor creation on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print("\nTest tensor successfully created on:", x.device)
        except Exception as e:
            print("Error creating tensor on GPU:", str(e))
    else:
        print("\nNo CUDA device available. Using CPU only.")

if __name__ == "__main__":
    check_cuda()