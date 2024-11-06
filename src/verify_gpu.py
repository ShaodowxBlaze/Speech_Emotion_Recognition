import torch

def verify_gpu():
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print("-" * 50)
        current_device = torch.cuda.current_device()
        print(f"Current Device: {current_device}")
        print(f"Device Name: {torch.cuda.get_device_name(current_device)}")
        print(f"Device Memory (GB): {torch.cuda.get_device_properties(current_device).total_memory / (1024**3):.2f}")
        
        # Test CUDA memory allocation
        try:
            print("\nTesting CUDA Memory Allocation...")
            x = torch.rand(1000, 1000).cuda()  # Test allocation
            print("Successfully allocated tensor on GPU")
            del x  # Free memory
            torch.cuda.empty_cache()  # Clear cache
        except Exception as e:
            print(f"Error allocating memory on GPU: {str(e)}")

if __name__ == "__main__":
    verify_gpu()