import subprocess
import sys

def check_gpu():
    try:
        # For Windows
        if sys.platform == 'win32':
            output = subprocess.check_output(['wmic', 'path', 'win32_VideoController', 'get', 'name'], text=True)
            return output
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    print("Checking GPU hardware...")
    print(check_gpu())