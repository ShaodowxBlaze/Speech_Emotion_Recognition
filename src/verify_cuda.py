import torch

def verify_cuda_setup():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA with a simple operation
        x = torch.rand(5, 3)
        print("\nTesting CUDA Operations:")
        print("Tensor created on CPU:", x.device)
        x = x.cuda()
        print("Tensor moved to GPU:", x.device)
        
        # Test GPU memory
        print("\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    verify_cuda_setup()