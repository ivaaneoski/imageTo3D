# Local CPU-Only Image-to-3D Relief Pipeline Tool

A local, CPU-only CLI tool that converts a single 2D image into a textured 3D mesh (`.obj`/`.glb`) via depth-relief geometry extraction. 

The pipeline performs:
`Image → Monocular Depth Estimation → Point Cloud Back-projection → Surface Reconstruction → Mesh Export`

---

## ⚠️ Known Limitations & Core Constraints (Read First)
To avoid overselling this tool, please be aware of the following physical and architectural limits:
1. **Relief Geometry Only (No Back-Side):** This tool does NOT perform "walk-around" full 3D object generation (which requires GPU-heavy diffusion models). It produces accurate depth and geometry for **what the camera saw only**. The back of the objects is open, and occluded geometry is not reconstructed.
2. **Relative Depth, Not Metric:** The estimated depth is relative (ordinal), not absolute. The dimensions of the output mesh do not correspond to physical real-world units (meters) unless you override camera intrinsics and define reference distances.
3. **Reflective & Textureless Surfaces:** Glass, mirrors, metallic objects, or completely featureless/flat surfaces will produce noisy or incorrect depth estimations. This is an inherent limitation of monocular depth models.
4. **Reconstruction Tradeoffs:**
   - **Poisson Surface Reconstruction (Default):** Produces smooth, connected meshes that handle noisy depth maps well. However, it can "balloon" edges or smooth out sharp details.
   - **Ball Pivoting Algorithm:** Preserves sharp geometric edges and is highly detailed, but it is extremely sensitive to noise and point density, often leaving holes in the mesh if points are too sparse.

---

## 🛠️ Installation & Setup

This tool is designed to run locally on a normal PC CPU and requires Python 3.11+.

1. **Clone or navigate to the repository directory:**
   ```bash
   cd d:/Programming/imageTo3D
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows (Command Prompt):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies (CPU-only PyTorch setup):**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 CLI Usage

Run the pipeline on an image with a single command:

```bash
python cli.py --input <path_to_image> --output <path_to_mesh.glb> [options]
```

### Basic Example (Poisson Reconstruction):
```bash
python cli.py --input test_data/input.jpg --output test_data/final_mesh_poisson.glb --model vits --method poisson
```

### Basic Example (Ball Pivoting):
```bash
python cli.py --input test_data/input.jpg --output test_data/final_mesh_bp.glb --model vits --method ball_pivoting
```

### Advanced Configuration Options:
- `--model {vits, vitb}`: Model checkpoint size for Depth Anything V2. Small (`vits`, default) or Base (`vitb`, higher RAM usage).
- `--no-onnx`: Disables ONNX Runtime inference acceleration and forces raw PyTorch on CPU.
- `--method {poisson, ball_pivoting}`: Reconstruction method to convert points to triangles. Default: `poisson`.
- `--fov <float>`: Assumed horizontal camera Field of View in degrees. Adjust this to stretch/compress depth perspective. Default: `60.0`.
- `--min-depth <float>`: Minimum target depth distance in meters. Default: `0.5`.
- `--max-depth <float>`: Maximum target depth distance in meters. Default: `2.0`.
- `--depth-mapping {inverse, linear}`: Mapping from normalized relative depth `[0, 1]` to metric coordinates. Default: `inverse` (disparity-based, mathematically accurate for disparity models).
- `--trim-percentile <float>`: Percentage of low-confidence boundary vertices to prune from the Poisson mesh. Default: `10.0`. Set to `0.0` to keep the entire mesh.
- `--poisson-depth <int>`: Octree resolution depth for Poisson reconstruction. Default: `9`.
- `--decimate <int>`: Target number of triangles for mesh decimation. Excellent for reducing GLB/OBJ file size for web rendering.
- `--save-intermediates`: Saves the intermediate grayscale depth map (`depth_map.png`) and point cloud (`point_cloud.ply`) in the designated folder.
- `--intermediates-dir <path>`: Folder path for intermediate files. Default: `intermediates`.

---

## 📂 Project Structure

```
image-to-3d/
├── src/
│   ├── __init__.py
│   ├── depth.py          # Depth Anything V2 loader & execution (ONNX / PyTorch)
│   ├── pointcloud.py     # Pinhole camera vectorized depth projection to 3D
│   ├── mesh.py           # Normals estimation & mesh reconstruction (Poisson / BPA)
│   ├── export.py         # Mesh exporter (OBJ and GLB formats)
│   └── pipeline.py       # Orchestrator combining stages 2-5
├── cli.py                # Pipeline entrypoint
├── requirements.txt      # Project library list
└── README.md             # This documentation file
```
