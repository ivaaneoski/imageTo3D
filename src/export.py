import os
import open3d as o3d

def export_mesh(mesh: o3d.geometry.TriangleMesh, output_path: str):
    """
    Exports a TriangleMesh to the specified format (.obj, .glb, etc.).
    
    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to export.
        output_path (str): File path to save the mesh.
    """
    if output_path.lower().endswith(".glb"):
        from pygltflib.utils import gltf2glb
        
        temp_gltf = output_path.replace(".glb", "_temp.gltf")
        temp_bin = output_path.replace(".glb", "_temp.bin")
        
        print(f"Exporting intermediate GLTF to {temp_gltf}...")
        success = o3d.io.write_triangle_mesh(temp_gltf, mesh)
        if not success:
            raise RuntimeError(f"Open3D failed to write intermediate GLTF mesh to {temp_gltf}")
            
        if os.path.exists(output_path):
            os.remove(output_path)
            
        print(f"Packaging GLTF to self-contained GLB: {output_path}...")
        try:
            gltf2glb(temp_gltf, output_path)
        finally:
            if os.path.exists(temp_gltf):
                os.remove(temp_gltf)
            if os.path.exists(temp_bin):
                os.remove(temp_bin)
    else:
        print(f"Exporting mesh to {output_path}...")
        success = o3d.io.write_triangle_mesh(output_path, mesh)
        if not success:
            raise RuntimeError(f"Open3D failed to write mesh to {output_path}")
            
    print(f"Successfully exported mesh to {output_path}")

def verify_glb(glb_path: str) -> bool:
    """
    Validates that a GLB file is valid by loading it back into Open3D 
    and checking that it has vertices and triangles.
    
    Args:
        glb_path (str): Path to the GLB file.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if not os.path.exists(glb_path):
        print(f"Validation failed: File {glb_path} does not exist.")
        return False
        
    try:
        # Read the mesh back from disk
        mesh = o3d.io.read_triangle_mesh(glb_path)
        num_vertices = len(mesh.vertices)
        num_triangles = len(mesh.triangles)
        
        print(f"Validation readback: Loaded {glb_path} successfully.")
        print(f"Mesh geometry check: {num_vertices} vertices, {num_triangles} triangles.")
        
        # Check that it actually has geometry
        if num_vertices == 0 or num_triangles == 0:
            print("Validation failed: Mesh contains zero vertices or triangles.")
            return False
            
        return True
    except Exception as e:
        print(f"Validation failed due to exception: {e}")
        return False
