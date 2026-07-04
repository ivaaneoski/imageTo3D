# Project: Image-to-3D Model Pipeline (CPU-only, local + GCP free tier)

## 1. What this is

A tool that converts a single 2D image into a textured 3D mesh (`.obj`/`.glb`) via:

```
Image → monocular depth estimation → point cloud → surface reconstruction → textured mesh
```

**Important constraint to design around:** this produces *relief geometry* — accurate depth for
what the camera saw, with no invented back-side or occluded geometry. It is NOT a
"walk-around" full-object reconstruction (that requires diffusion-based single-image-to-3D
models like TripoSR/InstantMesh, which need a GPU). This project should be scoped, named, and
documented honestly as a depth-relief mesh tool, not a full 3D object generator.

## 2. Hardware / deployment target

- Must run entirely on CPU. No CUDA dependency anywhere in the pipeline.
- Must run locally on a normal PC (no dedicated GPU assumed).
- Must also be deployable within Google Cloud's free tier (e2-micro or similar — think ~1-2 vCPU,
  ~1GB RAM on the always-free instance, or Cloud Run's free monthly quota). This means:
  - Model choice must default to the smallest viable checkpoint.
  - Memory footprint matters as much as speed — avoid loading multiple large models simultaneously.
  - Avoid anything that assumes >1GB RAM without explicit config to scale down.

## 3. Pipeline stages

### Stage 1 — Depth estimation
- Model: **Depth Anything V2**, small (`vits`) checkpoint by default; base (`vitb`) as an
  optional flag for users with more CPU/RAM headroom.
- Runtime: prefer **ONNX Runtime** over raw PyTorch for CPU inference speed. Structure the code
  so the model backend is swappable (PyTorch fallback if ONNX export isn't available for a
  given checkpoint).
- Input: arbitrary image (jpg/png), resized/padded to the model's expected input size.
- Output: a normalized relative depth map (this is NOT metric depth — no real-world units,
  no guaranteed consistent scale across different images). Document this clearly in code
  comments and README.

### Stage 2 — Depth map → point cloud
- Pure numpy vectorized back-projection using pinhole camera intrinsics:
  ```
  x = (u - cx) * d / fx
  y = (v - cy) * d / fy
  z = d
  ```
- Since there's no real camera calibration for an arbitrary input photo, use reasonable default
  intrinsics (e.g. assume a horizontal FOV like 60°, derive fx/fy/cx/cy from image dimensions).
  Expose these as CLI/config overrides for users who do know their camera's real FOV.
- Attach RGB color from the source image to each point (colored point cloud).
- Output: Open3D `PointCloud` object.

### Stage 3 — Point cloud → mesh
- Estimate normals (Open3D `estimate_normals`, consistent tangent plane orientation).
- Two reconstruction methods, selectable via flag:
  - **Poisson surface reconstruction** (default) — smoother, more robust to noisy depth.
  - **Ball pivoting** — preserves sharper detail, more sensitive to point density/noise.
- Post-process: remove low-density vertices from Poisson output (Open3D gives per-vertex
  density — trim the bottom percentile to avoid the "blobby" artifacts Poisson is known for
  around the mesh edges).
- Optional simplification/decimation step (`simplify_quadric_decimation`) to keep file sizes
  reasonable for a web viewer or download.

### Stage 4 — Texturing
- Project original image colors onto the mesh via vertex colors (simplest, already partially
  handled by carrying RGB through from Stage 2).
- Stretch goal only, not MVP: proper UV unwrapping + texture map baking. Flag this as a
  "phase 2" feature, don't block MVP on it — vertex coloring is good enough for a first version
  and dramatically simpler.

### Stage 5 — Export
- Export mesh to `.obj` (with `.mtl` if textured) and `.glb` (for easy web preview / AR quick-look).
- Open3D supports both natively (`write_triangle_mesh`).

## 4. Interfaces to build

Build in this order:

1. **CLI tool** (priority 1 — get this working end-to-end first)
   - `python cli.py --input photo.jpg --output model.glb --model vits --method poisson`
   - Flags for depth model size, reconstruction method, intrinsics overrides, decimation target.
2. **Local web UI** (priority 2)
   - Minimal FastAPI or Flask backend wrapping the same pipeline functions.
   - Simple upload-image → process → preview-and-download-mesh frontend.
   - Use `<model-viewer>` (Google's web component) or Three.js for in-browser mesh preview —
     both are lightweight and CPU-side (client renders, doesn't burden the server GPU-wise since
     there is none anyway).
3. **Cloud deployment config** (priority 3)
   - Dockerfile, CPU-only base image (no CUDA base images).
   - Deploy target: Cloud Run (scales to zero, free tier covers light usage) is a better fit
     than a persistent Compute Engine e2-micro, since e2-micro's RAM/CPU is very constrained
     and Cloud Run only bills while actively processing a request.
   - Keep cold-start in mind: lazy-load the depth model on first request, cache it in memory
     for the life of the container instance.

## 5. Tech stack

- Python 3.11+
- `torch` (CPU build) + `onnxruntime` for inference
- `transformers` (for loading Depth Anything V2 checkpoints from Hugging Face) or direct ONNX
  export if using a pre-exported model
- `open3d` for point cloud / mesh operations
- `numpy` for the back-projection math
- `Pillow` for image I/O
- FastAPI + Uvicorn for the web service (if built)
- Docker for the cloud deployment path

## 6. Repo structure (suggested)

```
image-to-3d/
├── src/
│   ├── depth.py          # depth estimation wrapper (model load + inference)
│   ├── pointcloud.py     # depth map -> Open3D point cloud
│   ├── mesh.py           # point cloud -> mesh (Poisson / ball pivoting)
│   ├── export.py         # mesh -> .obj/.glb
│   └── pipeline.py        # orchestrates the full flow, used by both CLI and API
├── cli.py
├── api/
│   ├── main.py           # FastAPI app
│   └── static/           # minimal frontend (upload + model-viewer preview)
├── Dockerfile
├── requirements.txt
└── README.md
```

## 7. Known limitations to document for users (be upfront about these)

- Output is relative depth, not metric — scale is not physically accurate unless the user
  supplies real camera intrinsics and a known reference distance.
- No occluded/back-side geometry — this is a relief mesh, not a complete 3D object.
- Reflective, transparent, or textureless surfaces will produce poor depth estimates
  (inherent limitation of monocular depth models, not a bug to chase).
- Poisson reconstruction can "balloon" or smooth out fine detail; ball pivoting is more
  detail-preserving but more sensitive to noisy/sparse points — mention both tradeoffs in README.

## 8. MVP definition (what "done" means for v1)

A user can run one CLI command on a photo and get back a `.glb` file that opens correctly in
a standard web-based 3D viewer, in well under a minute on a normal laptop CPU, using the small
Depth Anything V2 checkpoint and Poisson reconstruction as defaults.

Everything past that (web UI, cloud deploy, texture UV mapping, ball-pivoting polish) is
incremental and can be built after the MVP is proven end-to-end.