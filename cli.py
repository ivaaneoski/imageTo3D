import argparse
import sys
import os

# Configure stdout and stderr for UTF-8 to prevent Windows terminal codec encoding issues with emojis
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Add the workspace directory to the Python path to make imports work from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="CPU-only local 2D-image-to-3D relief model pipeline tool."
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
        help="Path structure for the output files (specifying 'model.glb' will yield 'model_raw.glb', 'model_filled.glb', and 'model_pointcloud.ply')."
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
    
    # Depth backprojection and intrinsics configuration
    parser.add_argument(
        "--method", 
        choices=["poisson", "ball_pivoting"], 
        default="poisson", 
        help="Reconstruction method: Poisson surface reconstruction or Ball Pivoting. Default: poisson."
    )
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
    
    # Mesh post-processing configuration
    parser.add_argument(
        "--trim-percentile", 
        type=float, 
        default=10.0, 
        help="For Poisson method: percentile of low-density vertices to trim. Set to 0 to disable. Default: 10.0."
    )
    parser.add_argument(
        "--poisson-depth", 
        type=int, 
        default=9, 
        help="For Poisson method: octree reconstruction depth. Default: 9."
    )
    parser.add_argument(
        "--decimate", 
        type=int, 
        default=None, 
        help="Optional: target number of triangles for mesh quadric decimation/simplification."
    )
    
    # Stage B repair configurations
    parser.add_argument(
        "--inpaint-threshold", 
        type=float, 
        default=0.1, 
        help="Depth jump threshold for negative outlier (pit) inpainting. Lower = more aggressive filling. Default: 0.1."
    )
    parser.add_argument(
        "--max-hole-size", 
        type=float, 
        default=0.1, 
        help="For repaired mesh: maximum hole boundary radius to fill (in meters). Default: 0.1."
    )
    
    # Output selection flags (Stage C)
    parser.add_argument(
        "--skip-raw", 
        action="store_true", 
        help="Skip saving the unrepaired raw mesh ('*_raw.glb')."
    )
    parser.add_argument(
        "--skip-filled", 
        action="store_true", 
        help="Skip saving the repaired mesh ('*_filled.glb')."
    )
    parser.add_argument(
        "--skip-pointcloud", 
        action="store_true", 
        help="Skip saving the raw point cloud ('*_pointcloud.ply')."
    )
    
    # Intermediates configuration
    parser.add_argument(
        "--save-intermediates", 
        action="store_true", 
        help="Save intermediate files (depth maps, point clouds) in the intermediates directory."
    )
    parser.add_argument(
        "--intermediates-dir", 
        default="intermediates", 
        help="Directory to save intermediate files. Default: intermediates."
    )
    
    # Optimization configurations
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable dynamic INT8 quantization of the ONNX model for CPU speedup."
    )
    parser.add_argument(
        "--use-ann-normals",
        action="store_true",
        help="Use SciPy's multi-threaded cKDTree for Approximate Nearest Neighbors normal estimation."
    )
    parser.add_argument(
        "--ann-eps",
        type=float,
        default=0.05,
        help="Error bound tolerance for Approximate Nearest Neighbors KD-Tree search. Default: 0.05."
    )
    parser.add_argument(
        "--no-fast-math",
        action="store_true",
        help="Disable CPU subnormal/denormal flushing and ONNX thread optimizations."
    )
    
    args = parser.parse_args()
    
    # Execute pipeline
    try:
        run_pipeline(
            image_path=args.input,
            output_path=args.output,
            model_size=args.model,
            method=args.method,
            fov_x=args.fov,
            d_min=args.min_depth,
            d_max=args.max_depth,
            mapping=args.depth_mapping,
            trim_percentile=args.trim_percentile,
            poisson_depth=args.poisson_depth,
            decimate_triangles=args.decimate,
            use_onnx=not args.no_onnx,
            save_intermediates=args.save_intermediates,
            intermediates_dir=args.intermediates_dir,
            save_raw=not args.skip_raw,
            save_filled=not args.skip_filled,
            save_pointcloud=not args.skip_pointcloud,
            inpaint_threshold=args.inpaint_threshold,
            max_hole_size=args.max_hole_size,
            use_quantize=args.quantize,
            use_ann_normals=args.use_ann_normals,
            ann_eps=args.ann_eps,
            use_fast_math=not args.no_fast_math
        )
    except Exception as e:
        print(f"\nPipeline failed with error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
