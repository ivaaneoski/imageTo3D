import numpy as np
import scipy.spatial as spatial
import open3d as o3d

def estimate_normals_ann(
    pcd: o3d.geometry.PointCloud,
    k: int = 30,
    eps: float = 0.05,
    camera_location: np.ndarray = np.array([0.0, 0.0, 0.0])
):
    """
    Estimates normals for the point cloud using SciPy's cKDTree approximate nearest neighbors search
    and vectorized Principal Component Analysis (PCA) covariance method.
    
    Args:
        pcd (o3d.geometry.PointCloud): Open3D PointCloud object to update.
        k (int): Number of neighbors. Default: 30.
        eps (float): Approximate search bound (non-negative). Default: 0.05.
        camera_location (np.ndarray): 3D coordinates of the camera to orient normals towards.
    """
    points = np.asarray(pcd.points)
    num_points = len(points)
    if num_points == 0:
        return
        
    # Guard against querying more neighbors than available points
    k = min(k, num_points)
    
    # 1. Build cKDTree
    tree = spatial.cKDTree(points)
    
    # 2. Query approximate nearest neighbors in parallel (using all CPU cores)
    # eps > 0 enables (1 + eps) approximate search which is significantly faster.
    # workers=-1 runs query across all CPU cores.
    _, indices = tree.query(points, k=k, eps=eps, workers=-1)
    
    # 3. Batch compute covariance matrices
    # Neighbors shape: (N, k, 3)
    neighbors = points[indices]
    
    # Mean of neighbors along axes: (N, 1, 3)
    mean = np.mean(neighbors, axis=1, keepdims=True)
    
    # Centered coordinates: (N, k, 3)
    centered = neighbors - mean
    
    # Covariance matrices: (N, 3, 3)
    # Matmul centered.transpose(0, 2, 1) [N, 3, k] and centered [N, k, 3]
    cov = np.matmul(centered.transpose(0, 2, 1), centered) / k
    
    # 4. Eigenvalue decomposition in batch mode
    # np.linalg.eigh returns eigenvalues in ascending order, and eigenvectors as columns.
    # Therefore, index 0 is the smallest eigenvalue, and column 0 is the normal vector.
    w, v = np.linalg.eigh(cov)
    
    # Extract the smallest eigenvector (normal vector): shape (N, 3)
    normals = v[:, :, 0]
    
    # 5. Orient normals consistently towards the camera location
    to_camera = camera_location - points
    dots = np.sum(normals * to_camera, axis=-1)
    flip_mask = dots < 0
    normals[flip_mask] = -normals[flip_mask]
    
    # 6. Normalize normal vectors to unit vectors
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    normals = normals / norms
    
    # Assign back to point cloud
    pcd.normals = o3d.utility.Vector3dVector(normals)
