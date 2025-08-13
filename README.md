# ComfyUI-MoGe-360: Complete 360° Layered 3D Pipeline

Transform 360° panoramas into layered 3D worlds with automatic object detection, high-quality matting, and spherical mesh generation.

## ✨ Features

### 🌐 All-in-One Complete Node
- **Pano360_To_Geometry_Complete**: Single node that takes a 360° panorama and outputs a complete 3D mesh
- **Quality Presets**: Fast, Balanced, High, and Ultra quality modes for different use cases
- **Optional Layering**: Enable/disable object detection and layer separation
- **Auto-Detection**: Open-vocabulary object detection with customizable queries
- **Material Presets**: Matte, metallic, glass material options for different scene types

### 🌐 Core Spherical MoGe Pipeline
- **360° Tile Sampler (Spherical)**: True spherical projection for equirectangular panoramas
- **MoGe Per-Tile Geometry**: Depth and normal estimation with proper camera intrinsics  
- **360° Depth/Normal Stitcher**: Seamless ERP reconstruction with seam continuity
- **360° Layer Mesher**: Multi-layer spherical mesh generation
- **Spherical Mesh Exporter**: GLB/PLY/OBJ export with proper UV mapping

### 🎯 Layered Processing System
- **OWL-ViT 360° Object Detector**: Open-vocabulary detection with ERP seam handling
- **Detection → ZIM Mattes**: High-quality object mask generation  
- **Sky/Background Splitter**: Automatic layer separation
- **360° Layer Builder**: Multi-layer composition with proper depth ordering
- **Layer Completion**: ERP-safe inpainting to fill occluded regions
- **Alpha Refinement**: Edge smoothing and mask cleanup

### ✅ Recent Fixes
- OWL‑ViT NMS now uses `[x,y,w,h]` box format to prevent OpenCV NMS errors.
- Detection preview clamps box coordinates to image bounds to avoid cv2 assertions.
- Spherical mesher depth normalization gains a robust fallback so terrain isn’t lost (sky‑only meshes fixed).

## 🚀 Quick Start

For the fastest experience, use the **Pano360_To_Geometry_Complete** node:

1. Load a 360° panorama image (2:1 aspect ratio)
2. Load a MoGe model using the "Load MoGe Model" node
3. Connect both to the complete node
4. Set quality preset (start with "balanced")
5. Enable layering for more sophisticated results (optional)
6. Execute to get a complete 3D mesh with textures

## Installation

1. Clone to your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cedarconnor/ComfyUI-MoGe-360.git
```

2. Install dependencies:
```bash
pip install -r ComfyUI-MoGe-360/requirements.txt
```

## Model Setup

### Required Models

#### MoGe Models (Auto-downloaded)
Place these in `ComfyUI/models/MoGe/`:
- Models from: https://huggingface.co/Kijai/MoGe_safetensors
- Automatically downloaded when using the "Load MoGe Model" node

#### OWL-ViT Models (For layered processing)
No manual setup needed. The node automatically downloads OWL‑ViT v1 (`google/owlvit-base-patch32`) via Hugging Face on first use. A local transformers cache under `models/transformers_cache/` will be used.

#### ZIM Matting Models (For layered processing)
Place these in `ComfyUI/models/zim/`:
- Download: `zim_vit_l_384.onnx` from https://huggingface.co/spaces/naver-ai/ZIM/tree/main
- Model size: ~1.3GB

### Model Directory Structure
```
ComfyUI/
├── models/
│   ├── MoGe/              # Auto-downloaded MoGe models
│   └── transformers_cache/ # Auto-downloaded OWL-ViT and other HF models
```

**Note**: OWL-ViT models are automatically downloaded from Hugging Face when first used.

## 🚀 Usage

Three complete workflows are provided in the `workflows/` folder:

1. **`quick_360_test.json`** - Fast testing with optimized settings (~5-8 minutes)
2. **`detection_pipeline_test.json`** - Focus on object detection and masking
3. **`complete_360_layered_pipeline.json`** - Full high-quality layered pipeline

### 🎯 Important: OWL-ViT Detection Settings

OWL‑ViT produces low confidence scores (typically 0.01–0.15). The default confidence threshold is **0.01**. Adjust between **0.001–0.1** for sensitivity control. Scores above 0.05 are relatively confident.

Note: NMS requires boxes in `[x, y, w, h]` (OpenCV `cv2.dnn.NMSBoxes`). This repo converts automatically; no action needed.

### 🧩 All‑in‑One Node

- Node: `Pano360_To_Geometry_Complete` under category `MoGe360/Complete`
- Inputs: ERP image, pre‑loaded MoGe model, quality preset, mesh resolution, layering toggle, and optional detection/layer params
- Outputs: TRIMESH mesh, depth/normal visualizations, process preview, progress report
- Presets: `fast`, `balanced`, `high`, `ultra` configure tile grid, MoGe levels, and stitching defaults

#### Advanced Options
- `layer_separation_mode`: `auto` | `sky_only` | `objects_only` (couples with layer selection during meshing)
- Grid overrides: `grid_yaw_override`, `grid_pitch_override`, `tile_resolution_override`, `stitcher_height_override`
- Export: `export_format` = none|glb|ply, with `export_prefix` path under ComfyUI output

### Basic Spherical MoGe Workflow

1. **Load 360° Panorama** → Input equirectangular image (2:1 aspect ratio)
2. **Load MoGe Model** → Choose model and precision (fp16 recommended)
3. **360° Tile Sampler** → Sample perspective tiles with spherical projection
4. **MoGe Per-Tile Geometry** → Process each tile for depth/normals
5. **360° Depth/Normal Stitcher** → Reconstruct full ERP depth map
6. **360° Layer Mesher** → Generate spherical mesh
7. **Spherical Mesh Exporter** → Export to GLB/PLY/OBJ

## ⚙️ Configuration

### Speed vs Quality Settings

**Fast (3-5 minutes)**:
- Tile Grid: 3×2, Tile Size: 512px, MoGe Resolution: 7
- Use `quick_360_test.json` workflow

**High Quality (15-20 minutes)**:
- Tile Grid: 6×3, Tile Size: 768px, MoGe Resolution: 8-9
- Use `complete_360_layered_pipeline.json` workflow

### Detection Parameters
- **confidence_threshold**: 0.001-0.1 (default: 0.01)
- **Text queries**: "mountain peak, rock formation, tree, building, person"
- **ERP mode**: "circular_padding" (handles ±180° seam) or "perspective_tiles" (thorough)
- **NMS threshold**: 0.5 (removes duplicate detections)

## 🛠️ Technical Details

### Spherical Projection System
- True spherical camera model replacing pinhole assumptions  
- Proper equirectangular (2:1) panorama handling
- ERP seam continuity across ±180° longitude boundary
- Pole-aware sampling and stitching

### ComfyUI Embedded Python

For local testing, prefer the embedded ComfyUI Python at `C:\\ComfyUI\\.venv\\Scripts\\python.exe` to ensure dependency alignment with ComfyUI.

### Memory Requirements
- **Minimum**: 6GB VRAM (3×2 tiles, 512px)
- **Recommended**: 12GB+ VRAM (6×3 tiles, 768px)
- Automatic batching prevents out-of-memory errors

### ERP Seam Handling
- **Circular padding** for detection across longitude seams
- **Weighted blending** in tile overlap regions
- **Mask wrapping** for objects crossing ±180°

## 🧯 Troubleshooting
- Geometry only shows sky:
  - Set `enable_layering=False` in `Pano360_To_Geometry_Complete` to isolate core spherical MoGe.
  - Use Balanced preset defaults: grid `6×3`, FOV `100°`, overlap `15°`, tile size `768`, output height `1024`.
  - The mesher now includes a robust depth normalization fallback; update and retry.
- OpenCV errors during detection/matting:
  - Ensure you run with ComfyUI’s embedded Python: `C:\\ComfyUI\\.venv\\Scripts\\python.exe`.
  - The detector clamps visualization boxes and uses correct NMS format; update and retry.
  - If errors persist, switch `erp_mode` to `direct` to bisect circular padding issues, then back to `circular_padding`.

## 🎥 Example Results

![Spherical MoGe Example](example_workflows/moge_example.png)

*360° panorama → Layered 3D mesh with sky, background, and object separation*
