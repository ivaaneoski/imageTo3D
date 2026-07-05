import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import onnxruntime as ort

# Map CLI size names to Hugging Face model IDs
MODEL_IDS = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf"
}

def get_model_path(model_size: str) -> str:
    """Returns the local path to cache the exported ONNX model."""
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(workspace_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    return os.path.join(weights_dir, f"depth_anything_v2_{model_size}.onnx")

def export_to_onnx(model_id: str, onnx_path: str):
    """Loads PyTorch model and exports it to ONNX format."""
    print(f"Exporting PyTorch model to ONNX: {model_id} -> {onnx_path}...")
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    
    # Depth Anything V2 models typically take 518x518 or 378x378 inputs.
    # We export with 518x518 dummy input and mark height/width as dynamic.
    dummy_input = torch.randn(1, 3, 518, 518)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
            "predicted_depth": {0: "batch_size", 1: "height", 2: "width"}
        },
        opset_version=14
    )
    print("ONNX export completed successfully.")

def estimate_depth(
    image_path: str,
    model_size: str = "vits",
    use_onnx: bool = True,
    use_quantize: bool = False,
    use_fast_math: bool = True
) -> np.ndarray:
    """
    Estimates depth for the input image using Depth Anything V2.
    Returns:
        np.ndarray: Normalized depth map (values between 0.0 and 1.0, 
                    where 1.0 is closest and 0.0 is farthest).
    """
    model_id = MODEL_IDS.get(model_size)
    if not model_id:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(MODEL_IDS.keys())}")
        
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # Load processor
    processor = AutoImageProcessor.from_pretrained(model_id)
    
    # Attempt ONNX inference if requested
    if use_onnx:
        onnx_path = get_model_path(model_size)
        try:
            if not os.path.exists(onnx_path):
                export_to_onnx(model_id, onnx_path)
            
            if use_quantize:
                from src.quantization import get_quantized_model
                model_to_run = get_quantized_model(onnx_path)
            else:
                model_to_run = onnx_path
                
            print(f"Running ONNX inference using model: {model_to_run}...")
            # Force size to exactly 518x518 to match the exported ONNX model's expected shape.
            # This prevents shape mismatch inside the vision transformer's reshape layers.
            image_square = image.resize((518, 518), resample=Image.Resampling.BILINEAR)
            inputs = processor(images=image_square, return_tensors="np")
            
            # Start ONNX session with optional CPU fast math thread optimizations
            if use_fast_math:
                from src.fast_math import get_optimized_session_options
                sess_options = get_optimized_session_options()
            else:
                sess_options = ort.SessionOptions()
                
            session = ort.InferenceSession(model_to_run, sess_options, providers=['CPUExecutionProvider'])
            ort_inputs = {session.get_inputs()[0].name: inputs["pixel_values"]}
            ort_outputs = session.run(None, ort_inputs)
            
            # Resizing outputs using PIL back to the original aspect ratio dimensions
            raw_depth = ort_outputs[0][0] # (H_model, W_model)
            depth_pil = Image.fromarray(raw_depth)
            depth_resized = depth_pil.resize((width, height), resample=Image.Resampling.BILINEAR)
            depth_arr = np.array(depth_resized, dtype=np.float32)
            
        except Exception as e:
            print(f"ONNX inference failed: {e}. Falling back to PyTorch.")
            use_onnx = False

    if not use_onnx:
        print(f"Running PyTorch inference using model: {model_id} on CPU...")
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model.eval()
        
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-process (resizes to original image size)
        prediction = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(height, width)]
        )[0]
        depth_arr = prediction["predicted_depth"].cpu().numpy()

    # Normalize depth map to [0, 1]
    depth_min = depth_arr.min()
    depth_max = depth_arr.max()
    if depth_max - depth_min > 1e-6:
        depth_normalized = (depth_arr - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_arr)
        
    return depth_normalized

# Viridis color map anchors for interpolation (avoiding matplotlib dependency)
VIRIDIS_ANCHORS = np.array([
    [0.267004, 0.004874, 0.329415],
    [0.282910, 0.116408, 0.469225],
    [0.252224, 0.233345, 0.528445],
    [0.193976, 0.344482, 0.554472],
    [0.143242, 0.448555, 0.556277],
    [0.108253, 0.548906, 0.548906],
    [0.137887, 0.650638, 0.518640],
    [0.264213, 0.751336, 0.433996],
    [0.477504, 0.821444, 0.318195],
    [0.720333, 0.871586, 0.178618],
    [0.993248, 0.906157, 0.143936]
], dtype=np.float32)

def colorize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """Colorizes a normalized [0, 1] depth map using the Viridis colormap."""
    indices = np.clip(depth_map, 0.0, 1.0) * (len(VIRIDIS_ANCHORS) - 1)
    idx_floor = np.floor(indices).astype(np.int32)
    idx_ceil = np.clip(idx_floor + 1, 0, len(VIRIDIS_ANCHORS) - 1)
    
    weight = (indices - idx_floor)[..., np.newaxis]
    color_map = (1.0 - weight) * VIRIDIS_ANCHORS[idx_floor] + weight * VIRIDIS_ANCHORS[idx_ceil]
    
    return (color_map * 255.0).astype(np.uint8)

def inpaint_depth_map(
    depth_map: np.ndarray,
    outlier_threshold: float = 0.1,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Detects sharp negative depth outliers (such as pits) using a local median filter
    and inpaints them using OpenCV's inpainting algorithm.
    """
    import cv2
    
    # Scale depth map to [0, 255] uint8 for OpenCV compatibility
    depth_8bit = (depth_map * 255.0).astype(np.uint8)
    
    # Run median filter to estimate local median
    local_median = cv2.medianBlur(depth_8bit, kernel_size)
    
    # We only want to fill "pits" (regions where depth is significantly lower/farther 
    # than local surroundings). Thus we check: local_median - depth_8bit > threshold.
    threshold_val = int(outlier_threshold * 255.0)
    diff = cv2.subtract(local_median, depth_8bit) # Saturation subtraction (val - depth)
    
    # Mask of outlier pixels (non-zero where outliers are present)
    _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
    
    if mask.sum() > 0:
        print(f"Inpainting {mask.sum() // 255} outlier depth pixels...")
        inpainted_8bit = cv2.inpaint(depth_8bit, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        inpainted_depth = inpainted_8bit.astype(np.float32) / 255.0
        return inpainted_depth
        
    return depth_map

def save_depth_visualization(depth_map: np.ndarray, output_path: str, colorized: bool = False):
    """Saves the normalized depth map as a grayscale or colorized PNG (brighter = closer)."""
    if colorized:
        vis = colorize_depth_map(depth_map)
    else:
        vis = (depth_map * 255.0).astype(np.uint8)
    Image.fromarray(vis).save(output_path)
    print(f"Saved depth visualization to {output_path}")
