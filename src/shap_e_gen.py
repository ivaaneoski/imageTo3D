import os
import torch
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_ply
from PIL import Image
from rembg import remove

def remove_background(image: Image.Image) -> Image.Image:
    """
    Removes the background of a PIL Image using rembg (U2Net).
    Returns an RGBA image where the background is transparent.
    """
    print("[BackgroundRemoval] Isolating the foreground subject using rembg...")
    # remove() returns an RGBA image with the background transparent
    return remove(image)

def generate_shape_e_mesh(
    image_path: str,
    output_ply_path: str,
    num_steps: int = 32,
    guidance_scale: float = 3.0,
    save_intermediates: bool = False,
    intermediates_dir: str = "intermediates"
):
    """
    Generates a 3D mesh from an input image using OpenAI's Shap-E image-to-3D pipeline on CPU.
    Before generating, it automatically removes the background to isolate the subject.
    
    Args:
        image_path (str): Path to the input 2D image.
        output_ply_path (str): Path to write the output PLY mesh.
        num_steps (int): Number of inference steps. Lower values are faster on CPU. Default: 32.
        guidance_scale (float): Guidance scale parameter. Default: 3.0.
        save_intermediates (bool): Whether to save intermediate files (like background-removed image).
        intermediates_dir (str): Directory to save intermediate files.
    """
    # Manually pre-download all configuration files to avoid HF download bugs on Windows
    from huggingface_hub import hf_hub_download
    repo_id = "openai/shap-e-img2img"
    configs = [
        "model_index.json",
        "prior/config.json",
        "renderer/config.json",
        "image_encoder/config.json",
        "image_processor/preprocessor_config.json",
        "scheduler/scheduler_config.json",
        "shap_e_renderer/config.json",
    ]
    print("[Shap-E] Checking and pre-downloading model configuration files...")
    for config_file in configs:
        try:
            hf_hub_download(repo_id=repo_id, filename=config_file)
        except Exception as e:
            print(f"[Shap-E] Warning check failed for {config_file}: {e}")

    print(f"\n[Shap-E] Initializing Shap-E image-to-3D pipeline on CPU...")
    pipe = ShapEImg2ImgPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True
    )
    pipe = pipe.to("cpu")
    
    print(f"[Shap-E] Loading and preprocessing input image...")
    image = Image.open(image_path)
    
    # 1. Remove background using rembg
    image_nobg = remove_background(image)
    
    # Save the background-isolated image for debug if requested
    if save_intermediates:
        os.makedirs(intermediates_dir, exist_ok=True)
        nobg_path = os.path.join(intermediates_dir, "photo_nobg.png")
        image_nobg.save(nobg_path)
        print(f"[Shap-E] Saved background-removed intermediate image to {nobg_path}")
    
    # 2. Resize to 256x256 (Shap-E's expected square format)
    # Note: We convert to RGBA to preserve the alpha channel transparency.
    image_resized = image_nobg.resize((256, 256), resample=Image.Resampling.BILINEAR)
    
    print(f"[Shap-E] Running generative 3D diffusion on CPU (steps={num_steps}, guidance={guidance_scale})...")
    # Run pipeline with output_type="mesh"
    outputs = pipe(
        image_resized,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        frame_size=256,
        output_type="mesh"
    ).images
    
    print(f"[Shap-E] Exporting generated 3D shape to PLY format...")
    # Save the output
    export_to_ply(outputs[0], output_ply_path)
    print(f"[Shap-E] Mesh successfully saved to {output_ply_path}")
