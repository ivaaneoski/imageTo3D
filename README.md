# Local CPU-Only Image-to-3D Relief Pipeline Tool

A local, CPU-only CLI tool that converts a single 2D image into a textured 3D mesh (`.obj`/`.glb`) via depth-relief geometry extraction. 

The pipeline performing this conversion is:
`Image → Monocular Depth Estimation → Depth Inpainting → Point Cloud Projection → Mesh Reconstruction → Mesh Export`

---

## 📸 Test Visualizations & Screenshots

Below are placeholders for the visual comparisons of the input and the generated 3D outputs. *(You can add your screenshots here)*

### 1. Input Image vs. Estimated Depth Map
| Input Image (`photo.jpg`) | Colorized Depth Map |
| :---: | :---: |
| *[Placeholder: Input Photo]* | *[Placeholder: Depth Map (Viridis)]* |

### 2. Mesh Reconstruction (Raw vs. Repaired)
| Raw Mesh (`_raw.glb` with gap/hole) | Repaired Mesh (`_filled.glb` closed) |
| :---: | :---: |
| *[Placeholder: Raw Mesh]* | *[Placeholder: Filled Mesh]* |

---

## 🛠️ Installation & Virtual Environment Setup

This tool runs entirely on CPU (no GPU required) and is tested on Python 3.11+.

1. **Clone and navigate to the project directory:**
   ```bash
   git clone https://github.com/ivaaneoski/imageTo3D.git
   cd imageTo3D
   ```

2. **Initialize the virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **On PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **On Command Prompt (cmd):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **On Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

4. **Install all dependencies (installs CPU-only PyTorch and headless OpenCV):**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 CLI Usage & Executing the Pipeline

Once the virtual environment is activated, run the full pipeline using your test image:

### Run the full pipeline (saves raw, filled, and point cloud outputs):
```bash
python cli.py --input photo.jpg --output test_data/portrait.glb --model vits --method poisson --save-intermediates
```

### Options & Customization Flags:
*   `--model {vits, vitb}`: Model size for Depth Anything V2. Small (`vits`, default) or Base (`vitb`, higher RAM/CPU load but significantly more detailed depth maps).
*   `--method {poisson, ball_pivoting}`: Reconstruct method. Default: `poisson`.
*   `--decimate <int>`: Target number of triangles for mesh decimation (e.g. `--decimate 50000` to simplify the mesh).
*   `--inpaint-threshold <float>`: Sensitivity for negative outlier (pit) depth detection. Lower values fill more aggressively. Default: `0.1`.
*   `--max-hole-size <float>`: Maximum hole boundary radius to fill (in meters). Default: `0.1`.
*   `--save-intermediates`: Saves the intermediate depth maps and density ply files in the `intermediates/` directory.
*   `--skip-raw` / `--skip-filled` / `--skip-pointcloud`: Optional flags to skip saving specific output formats.

### Standalone Point-Cloud Exporter
For fast image-to-point-cloud generation without mesh reconstruction:
```bash
python pointcloud_only.py --input photo.jpg --output test_data/standalone_cloud.ply
```

---

## 👁️ Recommended 3D Viewer Extensions (VS Code)

To inspect the point clouds and meshes directly inside VS Code, install the following extensions:
1.  **3D Point Cloud and Mesh Visualizer** by *kleinicke* (ideal for opening and inspecting `.ply` point cloud files).
2.  **Super GLB Viewer** by *JessyLeite* (ideal for viewing `.glb` 3D model meshes).

---

## ⚠️ Key Features & Pipeline Enhancements Done (Stages A–E)

We implemented robust repairs to handle monocular depth estimation failures (such as holes punched in dark hair against high-contrast backgrounds):

1.  **Stage A (Diagnosis):** Created custom Viridis depth colorizers and density point cloud maps to isolate whether gaps originated from the depth estimation or trimming.
2.  **Stage B (Repair Pipeline):**
    *   **Outlier Inpainting:** Compares the depth map against a local median filter and runs OpenCV `cv2.inpaint` to fill "pits" (regions where depth drops incorrectly to background values).
    *   **Hole-Aware Trimming:** Implemented graph-neighborhood voting on `mesh.adjacency_list` to prevent Poisson density-trimming from pruning isolated interior vertices.
    *   **Island Cleanup:** Automatically detects and removes tiny floating/disconnected triangles using triangle clustering.
    *   **Tensor Mesh Hole-Filling:** Bridges remaining open boundaries using the `o3d.t.geometry.TriangleMesh.fill_holes` API.
3.  **Stage C (Multi-Model Output):** A single execution now yields three distinct models under `test_data/`:
    *   `*_raw.glb` — The mesh exactly as reconstructed, holes and all.
    *   `*_filled.glb` — The repaired watertight mesh.
    *   `*_pointcloud.ply` — The raw colored point cloud directly from projection.
4.  **Stage D (Standalone Exporter):** Built a lightweight, independent script `pointcloud_only.py` to fetch point clouds fast.

---

## ⚠️ Known Limitations & Core Constraints
- **Relief Geometry Only (No Back-Side):** This tool does NOT perform "walk-around" full 3D object generation. It produces accurate depth and geometry for **what the camera saw only**. The back of the objects is open, and occluded geometry is not reconstructed.
- **Relative Depth, Not Metric:** The estimated depth is relative (disparity-based), not absolute. The dimensions of the output mesh do not correspond to physical real-world units (meters) unless you define reference distances.
- **Reflective & Textureless Surfaces:** Glass, mirrors, metallic objects, or completely featureless/flat surfaces will produce noisy or incorrect depth estimations.
