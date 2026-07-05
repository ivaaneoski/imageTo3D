import numpy as np
import open3d as o3d
from PIL import Image

def backproject_depth(
    depth_map: np.ndarray,
    color_image: Image.Image,
    fov_x: float = 60.0,
    d_min: float = 0.5,
    d_max: float = 2.0,
    mapping: str = "inverse",
    fx: float = None,
    fy: float = None,
    cx: float = None,
    cy: float = None,
    mask: np.ndarray = None
) -> o3d.geometry.PointCloud:
    """
    Back-projects a normalized depth map into a 3D point cloud using pinhole camera intrinsics.
    
    Args:
        depth_map (np.ndarray): 2D normalized depth map of shape (H, W), values in [0, 1].
        color_image (PIL.Image): Corresponding RGB color image.
        fov_x (float): Horizontal Field of View in degrees (used if fx, fy are not provided).
        d_min (float): Near depth limit in meters.
        d_max (float): Far depth limit in meters.
        mapping (str): Depth mapping type, either 'inverse' (disparity-like) or 'linear'.
        fx (float, optional): Focal length in x. Derived from fov_x if None.
        fy (float, optional): Focal length in y. Set to fx if None.
        cx (float, optional): Principal point x. Set to W/2 if None.
        cy (float, optional): Principal point y. Set to H/2 if None.
        mask (np.ndarray, optional): 2D boolean mask of shape (H, W) where True represents foreground.
        
    Returns:
        o3d.geometry.PointCloud: Colored point cloud.
    """
    H, W = depth_map.shape
    
    # 1. Derive or set intrinsics
    if fx is None:
        fov_rad = np.deg2rad(fov_x)
        fx = W / (2.0 * np.tan(fov_rad / 2.0))
    if fy is None:
        fy = fx
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0
        
    # Ensure depth_map is clipped to [0, 1]
    d_norm = np.clip(depth_map, 0.0, 1.0)
    
    # 2. Map normalized relative depth to metric depth (Z)
    if mapping == "inverse":
        # Disparity-like mapping: Z is proportional to 1 / (d_norm * (1/d_min - 1/d_max) + 1/d_max)
        inv_min = 1.0 / max(d_min, 1e-6)
        inv_max = 1.0 / max(d_max, 1e-6)
        z = 1.0 / (d_norm * (inv_min - inv_max) + inv_max)
    elif mapping == "linear":
        # Linear mapping: Z increases linearly as depth value decreases
        z = d_max - d_norm * (d_max - d_min)
    else:
        raise ValueError(f"Unknown mapping type: {mapping}. Use 'inverse' or 'linear'.")
        
    # 3. Vectorized coordinate calculation
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z_flat = z.flatten()
    
    x = (u - cx) * z_flat / fx
    # Note: image coordinates start from top-left (v=0 is top).
    # To map to 3D space where positive Y is up, we flip the Y coordinates:
    y = -(v - cy) * z_flat / fy
    
    vertices = np.stack([x, y, z_flat], axis=-1)
    
    # 4. Extract and normalize colors from the RGB image
    color_rgb = color_image.convert("RGB")
    color_arr = np.array(color_rgb, dtype=np.float32) / 255.0
    colors = color_arr.reshape(-1, 3)
    
    # Apply foreground mask if provided
    if mask is not None:
        keep = mask.flatten()
        vertices = vertices[keep]
        colors = colors[keep]
        
    # 5. Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: str):
    """Saves the Open3D PointCloud to a file (e.g. .ply)."""
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved point cloud to {output_path}")
