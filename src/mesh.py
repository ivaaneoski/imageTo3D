import numpy as np
import open3d as o3d

def reconstruct_mesh(
    pcd: o3d.geometry.PointCloud,
    method: str = "poisson",
    poisson_depth: int = 9,
    trim_percentile: float = 10.0,
    decimate_triangles: int = None,
    save_density_path: str = None,
    hole_aware: bool = True,
    max_hole_size: float = 0.1,
    use_ann_normals: bool = False,
    ann_eps: float = 0.05
) -> o3d.geometry.TriangleMesh:
    """
    Reconstructs a 3D mesh from a point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        method (str): Reconstruction method ('poisson' or 'ball_pivoting').
        poisson_depth (int): Octree depth for Poisson reconstruction.
        trim_percentile (float): Percentile threshold to trim low-density vertices (for Poisson).
        decimate_triangles (int, optional): Target number of triangles for simplification.
        save_density_path (str, optional): File path to save density point cloud (Stage A debug).
        hole_aware (bool): If True, applies neighbor voting and island removal.
        max_hole_size (float): Maximum hole boundary radius to fill (in meters).
        use_ann_normals (bool): If True, uses fast ANN normal estimation.
        ann_eps (float): Error bound for KD-tree search.
        
    Returns:
        o3d.geometry.TriangleMesh: Reconstructed mesh.
    """
    # 1. Normal Estimation
    if use_ann_normals:
        print(f"Estimating point cloud normals using Approximate Nearest Neighbors (ANN) (eps={ann_eps})...")
        from src.ann_normals import estimate_normals_ann
        estimate_normals_ann(pcd, k=30, eps=ann_eps)
    else:
        print("Estimating point cloud normals using standard Open3D KDTree...")
        # Estimate normals using hybrid search (radius-based + max neighbors)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        # Orient normals consistently towards the camera location at [0, 0, 0]
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))
        
        # Ensure normals are unit vectors
        pcd.normalize_normals()

    # 2. Reconstruction
    if method == "poisson":
        print(f"Reconstructing mesh using Poisson Surface Reconstruction (depth={poisson_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
        
        # Export untrimmed density visualization point cloud (Stage A Debug)
        if save_density_path is not None:
            print(f"Exporting Poisson density visualization to {save_density_path}...")
            dens_arr = np.asarray(densities)
            dens_min = dens_arr.min()
            dens_max = dens_arr.max()
            if dens_max - dens_min > 1e-6:
                dens_norm = (dens_arr - dens_min) / (dens_max - dens_min)
            else:
                dens_norm = np.zeros_like(dens_arr)
            
            # Map normalized densities to colors using Viridis
            from src.depth import colorize_depth_map
            dens_norm_2d = dens_norm.reshape(-1, 1)
            colors_rgb = colorize_depth_map(dens_norm_2d).reshape(-1, 3).astype(np.float32) / 255.0
            
            density_pcd = o3d.geometry.PointCloud()
            density_pcd.points = mesh.vertices
            density_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            o3d.io.write_point_cloud(save_density_path, density_pcd)
        
        # Trim low-density boundary vertices if requested
        if trim_percentile > 0:
            print(f"Trimming low-density vertices (trim_percentile={trim_percentile}%)...")
            densities_arr = np.asarray(densities)
            threshold = np.percentile(densities_arr, trim_percentile)
            vertices_to_remove = densities_arr < threshold
            
            if hole_aware:
                print("Applying hole-aware vertex trimming using graph-based neighborhood check...")
                mesh.compute_adjacency_list()
                adj = mesh.adjacency_list
                
                new_vertices_to_remove = np.copy(vertices_to_remove)
                for i, neighbors in enumerate(adj):
                    if vertices_to_remove[i]:
                        if len(neighbors) > 0:
                            num_remove_neighbors = sum(1 for n in neighbors if vertices_to_remove[n])
                            if (num_remove_neighbors / len(neighbors)) < 0.5:
                                new_vertices_to_remove[i] = False
                vertices_to_remove = new_vertices_to_remove
                
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            if hole_aware:
                mesh = remove_mesh_islands(mesh)
                
            if max_hole_size is not None and max_hole_size > 0:
                mesh = fill_mesh_holes(mesh, max_hole_size)
            
    elif method == "ball_pivoting":
        print("Reconstructing mesh using Ball Pivoting Algorithm...")
        # Compute average spacing to determine pivot ball radii
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2.0, avg_dist * 4.0]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    else:
        raise ValueError(f"Unknown reconstruction method: {method}. Choose 'poisson' or 'ball_pivoting'.")

    # 3. Cleanup mesh geometry
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # 4. Optional Decimation
    if decimate_triangles is not None and decimate_triangles > 0:
        num_triangles = len(mesh.triangles)
        if num_triangles > decimate_triangles:
            print(f"Decimating mesh from {num_triangles} to target {decimate_triangles} triangles...")
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=decimate_triangles)
            # Cleanup again after decimation
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

    return mesh

def remove_mesh_islands(mesh: o3d.geometry.TriangleMesh, min_triangles_ratio: float = 0.02) -> o3d.geometry.TriangleMesh:
    """
    Removes disconnected floating mesh components (islands) keeping the main components
    that have a triangle count larger than min_triangles_ratio * total_triangles.
    """
    print("Running connected components island cleanup...")
    try:
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        total_triangles = len(mesh.triangles)
        if len(cluster_n_triangles) > 1 and total_triangles > 0:
            largest_cluster_idx = np.argmax(cluster_n_triangles)
            
            # Keep any cluster containing at least min_triangles_ratio of total triangles (or at least 100 triangles)
            threshold = max(100, int(min_triangles_ratio * total_triangles))
            keep_cluster_indices = np.where(cluster_n_triangles >= threshold)[0]
            
            # Ensure the largest cluster is always kept
            if largest_cluster_idx not in keep_cluster_indices:
                keep_cluster_indices = np.append(keep_cluster_indices, largest_cluster_idx)
                
            triangles_to_keep = np.isin(triangle_clusters, keep_cluster_indices)
            triangles_to_remove = ~triangles_to_keep
            
            num_removed = np.sum(triangles_to_remove)
            print(f"Removed {num_removed} floating/disconnected triangles (kept components with >= {threshold} triangles).")
            mesh.remove_triangles_by_mask(triangles_to_remove)
    except Exception as e:
        print(f"Warning: Island removal failed: {e}")
    return mesh

def fill_mesh_holes(mesh: o3d.geometry.TriangleMesh, max_hole_size: float) -> o3d.geometry.TriangleMesh:
    """Fills mesh holes using tensor-based TriangleMesh.fill_holes API."""
    if max_hole_size is None or max_hole_size <= 0:
        return mesh
        
    print(f"Filling mesh holes (max_hole_size={max_hole_size}m)...")
    try:
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        filled_t_mesh = t_mesh.fill_holes(hole_size=max_hole_size)
        filled_mesh = filled_t_mesh.to_legacy()
        filled_mesh.compute_vertex_normals()
        return filled_mesh
    except Exception as e:
        print(f"Warning: Hole filling failed: {e}")
        return mesh
