import os
from PIL import Image
from src.depth import estimate_depth, save_depth_visualization
from src.pointcloud import backproject_depth, save_point_cloud
from src.mesh import reconstruct_mesh
from src.export import export_mesh, verify_glb

def run_pipeline(
    image_path: str,
    output_path: str,
    model_size: str = "vits",
    method: str = "poisson",
    fov_x: float = 60.0,
    d_min: float = 0.5,
    d_max: float = 2.0,
    mapping: str = "inverse",
    trim_percentile: float = 10.0,
    poisson_depth: int = 9,
    decimate_triangles: int = None,
    use_onnx: bool = True,
    save_intermediates: bool = False,
    intermediates_dir: str = "intermediates",
    save_raw: bool = True,
    save_filled: bool = True,
    save_pointcloud: bool = True,
    inpaint_threshold: float = 0.1,
    max_hole_size: float = 0.1
):
    """
    Orchestrates the entire image-to-3D pipeline from depth estimation to mesh export.
    Generates raw, filled, and point cloud outputs in a single run.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    print("--- Starting Image-to-3D Pipeline ---")
    
    # 0. Set up naming structure
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    base_filename = os.path.basename(output_path)
    name_without_ext, ext = os.path.splitext(base_filename)
    if not ext:
        ext = ".glb"
        
    raw_output_path = os.path.join(output_dir, f"{name_without_ext}_raw{ext}")
    filled_output_path = os.path.join(output_dir, f"{name_without_ext}_filled{ext}")
    pointcloud_output_path = os.path.join(output_dir, f"{name_without_ext}_pointcloud.ply")

    # 1. Depth Estimation
    print(f"\n[Stage 1/4] Estimating depth using model checkpoint: '{model_size}'...")
    depth_map = estimate_depth(image_path, model_size=model_size, use_onnx=use_onnx)
    
    # 2. Statistical Inpainting (Stage B repair)
    if save_filled:
        print("\n[Stage B] Running depth map outlier detection and inpainting...")
        from src.depth import inpaint_depth_map
        depth_map_inpainted = inpaint_depth_map(
            depth_map=depth_map,
            outlier_threshold=inpaint_threshold,
            kernel_size=5
        )
    else:
        depth_map_inpainted = depth_map
        
    # Save debug intermediates if requested
    if save_intermediates:
        os.makedirs(intermediates_dir, exist_ok=True)
        save_depth_visualization(depth_map, os.path.join(intermediates_dir, "depth_map.png"))
        save_depth_visualization(depth_map, os.path.join(intermediates_dir, "depth_map_color.png"), colorized=True)
        if save_filled:
            save_depth_visualization(depth_map_inpainted, os.path.join(intermediates_dir, "depth_map_inpainted.png"))
            save_depth_visualization(depth_map_inpainted, os.path.join(intermediates_dir, "depth_map_inpainted_color.png"), colorized=True)

    # 3. Point Cloud Projection
    image = Image.open(image_path)
    
    # A. Project Raw Depth
    pcd_raw = None
    if save_raw or save_pointcloud:
        print("\n[Stage 2/4] Projecting raw depth to 3D point cloud...")
        pcd_raw = backproject_depth(
            depth_map=depth_map,
            color_image=image,
            fov_x=fov_x,
            d_min=d_min,
            d_max=d_max,
            mapping=mapping
        )
        
    # B. Project Inpainted Depth
    pcd_filled = None
    if save_filled:
        print("\n[Stage 2/4] Projecting inpainted depth to 3D point cloud...")
        pcd_filled = backproject_depth(
            depth_map=depth_map_inpainted,
            color_image=image,
            fov_x=fov_x,
            d_min=d_min,
            d_max=d_max,
            mapping=mapping
        )
        
    if save_intermediates:
        if pcd_raw is not None:
            save_point_cloud(pcd_raw, os.path.join(intermediates_dir, "point_cloud.ply"))
        if pcd_filled is not None:
            save_point_cloud(pcd_filled, os.path.join(intermediates_dir, "point_cloud_inpainted.ply"))

    # Export point cloud directly if requested
    if save_pointcloud and pcd_raw is not None:
        print(f"\n[Output 3/3] Exporting raw point cloud to {pointcloud_output_path}...")
        save_point_cloud(pcd_raw, pointcloud_output_path)

    # 4. Mesh Reconstruction & Export
    
    # A. Reconstruct Raw Mesh
    if save_raw and pcd_raw is not None:
        print("\n[Output 1/3] Performing raw mesh reconstruction (unrepaired)...")
        density_path = os.path.join(intermediates_dir, "poisson_densities_raw.ply") if save_intermediates else None
        mesh_raw = reconstruct_mesh(
            pcd=pcd_raw,
            method=method,
            poisson_depth=poisson_depth,
            trim_percentile=trim_percentile,
            decimate_triangles=decimate_triangles,
            save_density_path=density_path,
            hole_aware=False,
            max_hole_size=0.0
        )
        export_mesh(mesh_raw, raw_output_path)
        if raw_output_path.lower().endswith(".glb"):
            verify_glb(raw_output_path)

    # B. Reconstruct Repaired Mesh (with Stage B fixes)
    if save_filled and pcd_filled is not None:
        print("\n[Output 2/3] Performing filled mesh reconstruction (repaired)...")
        density_path = os.path.join(intermediates_dir, "poisson_densities_filled.ply") if save_intermediates else None
        mesh_filled = reconstruct_mesh(
            pcd=pcd_filled,
            method=method,
            poisson_depth=poisson_depth,
            trim_percentile=trim_percentile,
            decimate_triangles=decimate_triangles,
            save_density_path=density_path,
            hole_aware=True,
            max_hole_size=max_hole_size
        )
        export_mesh(mesh_filled, filled_output_path)
        if filled_output_path.lower().endswith(".glb"):
            verify_glb(filled_output_path)

    print("\n--- Pipeline Completed Successfully ---")
