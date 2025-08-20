"""
Simplified ComfyUI-MoGe-360: Essential nodes only
Only includes the MoGe model loader and all-in-one complete node
"""

import os
import torch
import trimesh
import numpy as np
from PIL import Image
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Core MoGe imports
from .moge.model import MoGeModel
from .moge.utils.vis import colorize_depth
from .utils3d.numpy import image_mesh, image_uv, depth_edge

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False

import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths

# Configure only this module's logger instead of global logging
log = logging.getLogger(__name__)
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

# =============================================================================
# MoGe Model Loader Node
# =============================================================================

class DownloadAndLoadMoGeModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [   
                        "MoGe_ViT_L_fp16.safetensors",
                        "MoGe_ViT_L_fp32.safetensors",
                    ],
                    {"tooltip": "Downloads from 'https://huggingface.co/Kijai/MoGe_safetensors' to 'models/MoGe'"},
                ),
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp32", "tooltip": "The precision to use for the model weights"}),
            },
        }

    RETURN_TYPES = ("MOGEMODEL",)
    RETURN_NAMES = ("moge_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "MoGe360"
    DESCRIPTION = "Downloads and loads the selected MoGe model from Huggingface"

    def loadmodel(self, model, precision):
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_download_path = os.path.join(folder_paths.models_dir, 'MoGe')
        model_path = os.path.join(model_download_path, model)
   
        repo_id = "kijai/MoGE_safetensors"
        
        if not os.path.exists(model_path):
            log.info(f"Downloading moge model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{model}*"],
                local_dir=model_download_path,
                local_dir_use_symlinks=False,
            )
            
        model_config = {
            'encoder': 'dinov2_vitl14', 
            'remap_output': 'exp', 
            'output_mask': True, 
            'split_head': True, 
            'intermediate_layers': 4, 
            'dim_upsample': [256, 128, 64], 
            'dim_times_res_block_hidden': 2, 
            'num_res_blocks': 2, 
            'trained_area_range': [250000, 500000], 
            'last_conv_channels': 32, 
            'last_conv_size': 1
        }
        
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            model = MoGeModel(**model_config)
            
        model_sd = load_torch_file(model_path)
        
        if is_accelerate_available:
            for key in model_sd:
                set_module_tensor_to_device(model, key, dtype=dtype, device=device, value=model_sd[key])
        else:
            model.load_state_dict(model_sd, strict=True)
            model.to(dtype).to(device)
            
        model.eval()
        del model_sd

        return (model,)

# =============================================================================
# Simple All-in-One Node (without complex layering)
# =============================================================================

class Pano360ToGeometrySimple:
    """Simplified 360° panorama to 3D geometry conversion in a single node."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "erp_image": ("IMAGE", {
                    "tooltip": "Input equirectangular panorama image in 2:1 aspect ratio"
                }),
                "moge_model": ("MOGEMODEL", {
                    "tooltip": "Pre-loaded MoGe model for depth estimation"
                }),
                "quality": (["fast", "balanced", "high"], {
                    "default": "balanced",
                    "tooltip": "Processing quality: fast=quick preview, balanced=good quality, high=detailed"
                }),
                "mesh_resolution": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 128,
                    "tooltip": "Final mesh resolution. Higher = more detail but slower"
                }),
                "depth_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Depth scaling factor. 1.0=original, >1.0=exaggerated depth"
                }),
                "remove_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove mesh artifacts at depth discontinuities"
                }),
            },
            "optional": {
                "export_format": (["none", "glb", "ply"], {
                    "default": "none",
                    "tooltip": "Export mesh to disk"
                }),
                "export_prefix": ("STRING", {
                    "default": "3D/MoGe360",
                    "tooltip": "Export filename prefix"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "STRING")
    RETURN_NAMES = ("mesh", "depth_preview", "progress_log")
    FUNCTION = "process"
    CATEGORY = "MoGe360"
    DESCRIPTION = "Complete 360° panorama to 3D mesh conversion (simplified version)"

    def process(self, erp_image, moge_model, quality, mesh_resolution, depth_scale, remove_edges,
                export_format="none", export_prefix="3D/MoGe360"):
        
        device = mm.get_torch_device()
        start_time = time.time()
        
        progress_log = []
        progress_log.append("=== MoGe360 Simple Pipeline ===")
        progress_log.append(f"Input: {erp_image.shape}")
        progress_log.append(f"Quality: {quality}")
        
        try:
            # Validate input
            if not self._validate_input(erp_image):
                raise ValueError("Invalid input: Expected 2:1 aspect ratio ERP image")
            
            # Configure quality settings
            settings = self._get_quality_settings(quality)
            progress_log.append(f"Settings: {settings}")
            
            # Process through MoGe
            progress_log.append("Running MoGe depth estimation...")
            
            B, H, W, C = erp_image.shape
            input_tensor = erp_image.permute(0, 3, 1, 2).to(device)
            
            # Run MoGe inference
            with torch.no_grad():
                output = moge_model.infer(input_tensor[0], 
                                        resolution_level=settings["resolution_level"], 
                                        apply_mask=True)
            
            points_tensor = output['points']
            depth_tensor = output['depth']
            mask_tensor = output['mask']
            
            # Convert to numpy
            points_np = points_tensor.cpu().numpy()
            depth_np = depth_tensor.cpu().numpy()
            mask_np = mask_tensor.cpu().numpy()
            input_np = erp_image.cpu().numpy().astype(np.float32)

            progress_log.append(f"Generated depth map: {depth_np.shape}")
            
            # Create mesh
            progress_log.append(f"Generating 3D mesh at {mesh_resolution} resolution...")
            
            # Use mesh_resolution to control UV grid density
            # Scale resolution to maintain aspect ratio
            aspect_ratio = W / H
            mesh_width = int(mesh_resolution * aspect_ratio)
            mesh_height = mesh_resolution
            
            faces, vertices, vertex_colors, vertex_uvs = image_mesh(
                points_np,
                input_np[0],
                image_uv(width=mesh_width, height=mesh_height),
                mask=mask_np & ~depth_edge(depth_np, mask=mask_np, rtol=0.02) if remove_edges else mask_np,
                tri=True
            )
            
            # Apply depth scaling and coordinate correction
            vertices = vertices * [depth_scale, -depth_scale, -depth_scale]
            vertex_uvs = vertex_uvs * [1, -1] + [0, 1]
            
            # Create trimesh object
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=trimesh.visual.texture.TextureVisuals(
                    uv=vertex_uvs,
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorTexture=Image.fromarray((input_np[0] * 255).astype(np.uint8)),
                        metallicFactor=0.0,
                        roughnessFactor=0.8
                    )
                ),
                process=False
            )
            
            progress_log.append(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
            
            # Create depth visualization
            depth_vis = colorize_depth(depth_np, mask=mask_np, normalize=True)
            depth_vis = torch.from_numpy(depth_vis).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).float() / 255.0
            
            # Export if requested
            if export_format != "none":
                try:
                    full_output_folder, filename, counter, subfolder, filename_prefix = \
                        folder_paths.get_save_image_path(export_prefix, folder_paths.get_output_directory())
                    
                    export_path = Path(full_output_folder) / f'{filename}_{counter:05d}_.{export_format}'
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    mesh.export(export_path)
                    progress_log.append(f"Exported: {export_path}")
                except Exception as e:
                    progress_log.append(f"Export failed: {e}")
            
            # Final stats
            total_time = time.time() - start_time
            progress_log.append(f"Completed in {total_time:.1f}s")
            
            return (mesh, depth_vis, "\n".join(progress_log))
            
        except Exception as e:
            import traceback
            error_time = time.time() - start_time
            stack_trace = traceback.format_exc()
            progress_log.append(f"ERROR after {error_time:.1f}s: {str(e)}")
            progress_log.append(f"Stack trace:\n{stack_trace}")
            log.error(f"Processing failed: {e}\n{stack_trace}")
            
            # Return empty results
            empty_mesh = trimesh.Trimesh()
            empty_img = torch.zeros((1, 512, 1024, 3), dtype=torch.float32)
            
            return (empty_mesh, empty_img, "\n".join(progress_log))

    def _validate_input(self, image):
        """Validate ERP input."""
        try:
            _, H, W, C = image.shape
            aspect_ratio = W / H
            return C == 3 and abs(aspect_ratio - 2.0) < 0.1
        except:
            return False

    def _get_quality_settings(self, quality):
        """Get processing settings for quality level."""
        settings = {
            "fast": {"resolution_level": 7},
            "balanced": {"resolution_level": 9}, 
            "high": {"resolution_level": 11}
        }
        return settings.get(quality, settings["balanced"])