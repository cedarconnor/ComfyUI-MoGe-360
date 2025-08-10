# ComfyUI-MoGe-360: Spherical MoGe for 360° Panoramas

Convert 360° panoramas into layered 3D worlds using spherical MoGe depth estimation, automatic object detection, and layer-aware 3D reconstruction.

## Features

### Core Spherical MoGe Pipeline
- **360° Tile Sampler (Spherical)**: Samples perspective tiles from equirectangular panoramas using true spherical projection
- **MoGe Per-Tile Geometry**: Processes tiles through MoGe model to get depth and normals with proper camera intrinsics  
- **360° Depth/Normal Stitcher**: Seamlessly stitches tile outputs back to equirectangular format
- **360° Layer Mesher**: Creates spherical meshes from layered depth and RGB data
- **Spherical Mesh Exporter**: Exports meshes to GLB/PLY/OBJ formats

### Layered Processing (Future)
- Object detection with OWL-ViT (open-vocabulary)
- ZIM matting for high-quality object masks
- Sky/background/foreground layer separation
- ERP-aware inpainting for layer completion
- Depth alignment across layers

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
Place these in `ComfyUI/models/owlvit/`:
- Download from: https://huggingface.co/google/owlv2-base-patch16-ensemble
- Required files: `pytorch_model.bin`, `config.json`, `preprocessor_config.json`

#### ZIM Matting Models (For layered processing)
Place these in `ComfyUI/models/zim/`:
- Download: `zim_vit_l_384.onnx` from https://huggingface.co/spaces/naver-ai/ZIM/tree/main
- Model size: ~1.3GB

### Model Directory Structure
```
ComfyUI/
├── models/
│   ├── MoGe/              # Auto-downloaded MoGe models
│   ├── owlvit/            # Manual: OWL-ViT detection models  
│   └── zim/               # Manual: ZIM matting models
```

## Usage

### Basic Spherical MoGe Workflow

1. **Load MoGe Model** → Choose model and precision
2. **360° Tile Sampler (Spherical)** → Input ERP panorama
3. **MoGe Per-Tile Geometry** → Process tiles for depth/normals
4. **360° Depth/Normal Stitcher** → Reconstruct ERP depth/normals
5. **360° Layer Mesher** → Create spherical mesh
6. **Spherical Mesh Exporter** → Export to GLB/PLY/OBJ

### Parameters

- **Tile Grid**: 6×3 recommended (6 yaw × 3 pitch)
- **Tile FOV**: 100° with 15° overlap for seamless stitching
- **Tile Resolution**: 768px (balance of quality vs speed)
- **Mesh Resolution**: 1024px for final spherical mesh

## Technical Details

### Spherical Projection
- Uses true spherical camera model instead of pinhole assumptions
- Handles equirectangular (2:1) panorama projection correctly
- ERP seam continuity across ±180° longitude boundary
- Proper pole handling for latitude extremes

### Memory Requirements
- Powerful GPU recommended for 6×3 tile grid processing
- ~8GB VRAM for 768px tiles with batch processing
- Automatic batching and memory management included

![example](example_workflows/moge_example.png)