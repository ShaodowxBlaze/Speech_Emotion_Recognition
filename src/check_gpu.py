import torch
import psutil
import os

def check_system_resources():
    print("\nSystem Resource Information:")
    print("-" * 50)
    
    # CPU Info
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Cores: {cpu_count}")
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory Info
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
    
    # GPU Info
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print("-" * 50)
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu.name}")
            print(f"Total Memory: {gpu.total_memory / (1024**2):.2f} MB")
            print(f"CUDA Version: {torch.version.cuda}")
            
            # Current GPU memory usage
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)
            memory_cached = torch.cuda.memory_reserved(i) / (1024**2)
            print(f"Memory Allocated: {memory_allocated:.2f} MB")
            print(f"Memory Cached: {memory_cached:.2f} MB")
    else:
        print("\nNo GPU available. Using CPU only.")

if __name__ == "__main__":
    check_system_resources()