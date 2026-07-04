import argparse
import sys
import os
from PIL import Image

# Configure stdout and stderr for UTF-8 to prevent Windows terminal codec encoding issues with emojis
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add the workspace directory to the Python path to make imports work from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.depth import estimate_depth
from src.pointcloud import backproject_depth, save_point_cloud

def main():
    parser = argparse.ArgumentParser(
        description="CPU-only local 2D-image-to-point-cloud fast exporter tool."
    )
    
    # Required Arguments
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to the input image (JPG or PNG)."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to the output point cloud file (e.g. cloud.ply)."
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        choices=["vits", "vitb"], 
        default="vits", 
        help="Depth Anything V2 model checkpoint size. Small ('vits') or Base ('vitb'). Default: vits."
    )
    parser.add_argument(
        "--no-onnx", 
        action="store_true", 
        help="Disable ONNX Runtime inference acceleration and force raw PyTorch on CPU."
    )
    
    # Intrinsics and Depth configurations
    parser.add_argument(
        "--fov", 
        type=float, 
        default=60.0, 
        help="Assumed camera horizontal Field of View in degrees for intrinsics calculation. Default: 60.0."
    )
    parser.add_argument(
        "--min-depth", 
        type=float, 
        default=0.5, 
        help="Target minimum (near clipping) depth mapping distance in meters. Default: 0.5."
    )
    parser.add_argument(
        "--max-depth", 
        type=float, 
        default=2.0, 
        help="Target maximum (far clipping) depth mapping distance in meters. Default: 2.0."
    )
    parser.add_argument(
        "--depth-mapping", 
        choices=["inverse", "linear"], 
        default="inverse", 
        help="Mapping type from normalized depth to metric depth. Default: inverse (disparity-based)."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found at {args.input}", file=sys.stderr)
        sys.exit(1)
        
    print("--- Fast Image-to-Point-Cloud Exporter ---")
    
    try:
        # 1. Depth Estimation
        print(f"\n[Stage 1/2] Estimating depth using model: '{args.model}'...")
        depth_map = estimate_depth(args.input, model_size=args.model, use_onnx=not args.no_onnx)
        
        # 2. Point Cloud Back-projection
        print("\n[Stage 2/2] Back-projecting depth to 3D point cloud...")
        image = Image.open(args.input)
        pcd = backproject_depth(
            depth_map=depth_map,
            color_image=image,
            fov_x=args.fov,
            d_min=args.min_depth,
            d_max=args.max_depth,
            mapping=args.depth_mapping
        )
        
        # 3. Export point cloud
        print(f"\nSaving point cloud to {args.output}...")
        save_point_cloud(pcd, args.output)
        
        print("\n--- Point Cloud Generation Completed Successfully ---")
    except Exception as e:
        print(f"\nError: Point cloud generation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
