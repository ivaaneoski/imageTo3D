import os
import onnxruntime.quantization as ort_quant

def get_quantized_model(float_model_path: str) -> str:
    """
    Given a path to a float32 ONNX model, checks if a quantized INT8 version exists.
    If not, it performs dynamic quantization and saves it in the same directory.
    
    Args:
        float_model_path (str): Absolute or relative path to the float32 ONNX model.
        
    Returns:
        str: Path to the quantized INT8 ONNX model.
    """
    if not os.path.exists(float_model_path):
        raise FileNotFoundError(f"Source float32 ONNX model not found: {float_model_path}")
        
    base, ext = os.path.splitext(float_model_path)
    quant_model_path = f"{base}_quantized{ext}"
    
    if not os.path.exists(quant_model_path):
        print(f"\n[Quantization] INT8 quantized model not found. Commencing dynamic quantization...")
        print(f"[Quantization] Input: {float_model_path}")
        print(f"[Quantization] Output: {quant_model_path}")
        
        # Execute dynamic quantization
        # QInt8 is suitable for general CPU deployments, as it quantizes weights to signed 8-bit integers.
        ort_quant.quantize_dynamic(
            model_input=float_model_path,
            model_output=quant_model_path,
            weight_type=ort_quant.QuantType.QInt8
        )
        print("[Quantization] Dynamic quantization completed successfully.")
    else:
        print(f"[Quantization] Using existing quantized model: {quant_model_path}")
        
    return quant_model_path
