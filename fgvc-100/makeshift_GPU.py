"""
GPU Monitoring Utility

This script provides a simple way to monitor GPU status and utilization
in real-time through the terminal. It uses the nvidia-smi command to
retrieve GPU information and displays it at regular intervals.

This was used to validate that the GPU was being utilized in the initial
training runs.

Usage:
    Run this script in the background while training models or performing
    GPU-intensive tasks to monitor resource usage.

Requirements:
    - NVIDIA GPU with drivers installed
    - nvidia-smi command available in system path
"""

import time
import subprocess

def print_gpu_status():
    """
    Retrieve and display current GPU status information.
    
    Uses the nvidia-smi command to get GPU utilization, memory usage,
    temperature, and other metrics. Prints the raw output to the console.
    
    Returns:
        None
    """
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode()
        print("\n----- GPU STATUS -----")
        print(output)
    except Exception as e:
        print("Could not retrieve GPU usage:", e)

if __name__ == "__main__":
    print("Monitoring GPU every 5 seconds. Press Ctrl+C to stop.")
    try:
        while True:
            print_gpu_status()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped GPU monitor.")
