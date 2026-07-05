import os
import torch
import onnxruntime as ort

def setup_fast_math():
    """
    Configures CPU-specific mathematical optimizations.
    Flushes denormal (subnormal) floating-point numbers to zero to prevent CPU microcode stalls.
    """
    try:
        # Denormals occur when values become extremely close to zero.
        # Handling them in hardware is slow. Flushing them to zero speeds up CPU math significantly.
        torch.set_flush_denormal(True)
        print("[FastMath] Subnormal/denormal CPU math flushing enabled.")
    except Exception as e:
        print(f"[FastMath] Warning: Could not enable denormal flushing: {e}")

def get_optimized_session_options() -> ort.SessionOptions:
    """
    Creates and returns optimized SessionOptions for ONNX Runtime CPU execution.
    
    Returns:
        ort.SessionOptions: Configured session options.
    """
    sess_options = ort.SessionOptions()
    
    # Enable all graph optimizations (constant folding, node fusions, etc.)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Configure intra-op threads to match physical core count for CPU efficiency.
    # Set to 0 to let ONNX Runtime auto-tune.
    cores = os.cpu_count()
    if cores:
        # Use physical cores if possible (heuristically half of logical processors)
        # to prevent thread thrashing / context switching overhead.
        physical_cores = max(1, cores // 2)
        sess_options.intra_op_num_threads = physical_cores
        print(f"[FastMath] Configuring ONNX Runtime intra-op threads: {physical_cores} (Logical CPUs: {cores})")
    else:
        sess_options.intra_op_num_threads = 0
        print("[FastMath] Configuring ONNX Runtime intra-op threads: Auto")
        
    return sess_options
