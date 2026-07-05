import numpy as np
import open3d as o3d

def create_double_sided_point_cloud(
    pcd: o3d.geometry.PointCloud,
    d_max: float,
    use_ann_normals: bool = False,
    ann_eps: float = 0.05,
    is_isolated: bool = False
) -> o3d.geometry.PointCloud:
    """
    Generates a double-sided point cloud by estimating normals on the front-side,
    mirroring the geometry and normals along the background plane or dynamic boundary plane,
    and merging them into a single watertight point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): Front-side point cloud.
        d_max (float): Far depth clipping distance in meters (acts as the symmetry plane if not is_isolated).
        use_ann_normals (bool): If True, uses ANN normal estimation for the front side.
        ann_eps (float): Search tolerance bound for ANN.
        is_isolated (bool): If True, computes a dynamic symmetry plane at the subject's boundary depth.
        
    Returns:
        o3d.geometry.PointCloud: Merged, double-sided point cloud with pre-computed outwards normals.
    """
    # 1. Estimate normals on the front point cloud first if they don't exist
    if not pcd.has_normals():
        print("[DoubleSided] Estimating normals on front point cloud before mirroring...")
        if use_ann_normals:
            from src.ann_normals import estimate_normals_ann
            estimate_normals_ann(pcd, k=30, eps=ann_eps)
        else:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
            pcd.normalize_normals()
            
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    
    if len(points) == 0:
        return pcd
        
    # 2. Determine the symmetry plane
    if is_isolated:
        # Use the 98th percentile of the Z coordinates as the symmetry plane
        symmetry_plane = np.percentile(points[:, 2], 98)
        print(f"[DoubleSided] Using dynamic isolated subject boundary plane at Z={symmetry_plane:.4f}m")
    else:
        symmetry_plane = d_max
        print(f"[DoubleSided] Using static background plane at Z={symmetry_plane:.4f}m")
        
    # 3. Mirror points: Z_back = 2 * symmetry_plane - Z
    mirrored_points = np.copy(points)
    mirrored_points[:, 2] = 2.0 * symmetry_plane - points[:, 2]
    
    # 4. Mirror normals: flip the Z component to make them point backwards
    mirrored_normals = np.copy(normals)
    mirrored_normals[:, 2] = -normals[:, 2]
    
    # 5. Slightly darken back-side colors to differentiate the sides
    mirrored_colors = colors * 0.85
    
    # 6. Merge front and back
    double_sided_points = np.concatenate([points, mirrored_points], axis=0)
    double_sided_colors = np.concatenate([colors, mirrored_colors], axis=0)
    double_sided_normals = np.concatenate([normals, mirrored_normals], axis=0)
    
    # Create merged PointCloud
    double_sided_pcd = o3d.geometry.PointCloud()
    double_sided_pcd.points = o3d.utility.Vector3dVector(double_sided_points)
    double_sided_pcd.colors = o3d.utility.Vector3dVector(double_sided_colors)
    double_sided_pcd.normals = o3d.utility.Vector3dVector(double_sided_normals)
    
    print(f"[DoubleSided] Fused front + back point clouds: {len(double_sided_points)} points total.")
    return double_sided_pcd
