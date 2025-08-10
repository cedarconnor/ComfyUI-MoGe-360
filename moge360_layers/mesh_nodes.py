"""
Spherical meshing nodes for converting depth/normals to 3D meshes.
Implements proper spherical projection for panoramic content.
"""

import os
import torch
import numpy as np
import trimesh
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

from .spherical_utils import SphericalProjection
from ..utils3d.numpy import image_mesh, image_uv, depth_edge
import comfy.model_management as mm
import folder_paths

log = logging.getLogger(__name__)

class Layer_Mesher_Spherical:
    """Create spherical meshes from layered panoramic data."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_stack": ("LAYER_STACK",),
                "mesh_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 128}),
                "layer_selection": (["all", "sky_only", "background_only", "objects_only"], {"default": "all"}),
                "depth_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "remove_edge": ("BOOLEAN", {"default": True}),
                "smooth_normals": ("BOOLEAN", {"default": True}),
                "metallic_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "roughness_factor": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_layers": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "layer_rgb": ("IMAGE",),  # Legacy single layer input
                "layer_depth": ("IMAGE",), 
                "layer_alpha": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE")
    RETURN_NAMES = ("spherical_mesh", "sphere_depth", "sphere_normals")
    FUNCTION = "create_spherical_mesh"
    CATEGORY = "MoGe360/Meshing"
    DESCRIPTION = "Create spherical meshes from layered panoramic data"

    def create_spherical_mesh(self, layer_stack=None, mesh_resolution: int = 1024, 
                            layer_selection: str = "all", depth_scale: float = 1.0,
                            remove_edge: bool = True, smooth_normals: bool = True, 
                            metallic_factor: float = 0.0, roughness_factor: float = 0.8,
                            merge_layers: bool = False, layer_rgb: torch.Tensor = None, 
                            layer_depth: torch.Tensor = None, layer_alpha: torch.Tensor = None):
        
        device = mm.get_torch_device()
        
        # Determine if using layered system or legacy mode
        if layer_stack is not None:
            return self._create_from_layer_stack(layer_stack, mesh_resolution, layer_selection, 
                                               depth_scale, remove_edge, smooth_normals, 
                                               metallic_factor, roughness_factor, merge_layers, device)
        
        # Legacy mode - single layer inputs
        if layer_rgb is None or layer_depth is None or layer_alpha is None:
            raise ValueError("Either layer_stack or all legacy inputs (layer_rgb, layer_depth, layer_alpha) must be provided")
        
        return self._create_legacy_inputs(layer_rgb, layer_depth, layer_alpha, mesh_resolution,
                                         depth_scale, remove_edge, smooth_normals, 
                                         metallic_factor, roughness_factor, device)
    
    def _create_from_layer_stack(self, layer_stack: Dict, mesh_resolution: int, layer_selection: str,
                                depth_scale: float, remove_edge: bool, smooth_normals: bool,
                                metallic_factor: float, roughness_factor: float, merge_layers: bool, device):
        """Create meshes from layered data."""
        
        layers = layer_stack['layers']
        H, W = layer_stack['dimensions']
        
        log.info(f"Creating meshes from {len(layers)} layers: {[l['type'] for l in layers]}")
        
        # Filter layers based on selection
        selected_layers = []
        for layer in layers:
            layer_type = layer['type']
            if (layer_selection == "all" or 
                (layer_selection == "sky_only" and layer_type == "sky") or
                (layer_selection == "background_only" and layer_type == "background") or
                (layer_selection == "objects_only" and layer_type == "object")):
                selected_layers.append(layer)
        
        if not selected_layers:
            log.warning(f"No layers selected with filter: {layer_selection}")
            # Return empty mesh
            empty_mesh = trimesh.Trimesh()
            empty_tensor = torch.zeros((1, H, W, 3), device=device)
            return (empty_mesh, empty_tensor, empty_tensor)
        
        log.info(f"Selected {len(selected_layers)} layers: {[l['type'] for l in selected_layers]}")
        
        if merge_layers or len(selected_layers) == 1:
            # Merge layers into single mesh
            return self._create_merged_mesh(selected_layers, mesh_resolution, depth_scale, 
                                          remove_edge, smooth_normals, metallic_factor, 
                                          roughness_factor, device, H, W)
        else:
            # For now, just use the first selected layer
            # TODO: Return multiple meshes in future
            return self._create_layer_mesh(selected_layers[0], mesh_resolution, depth_scale,
                                         remove_edge, smooth_normals, metallic_factor,
                                         roughness_factor, device, H, W)
    
    def _create_merged_mesh(self, layers: List[Dict], mesh_resolution: int, depth_scale: float,
                           remove_edge: bool, smooth_normals: bool, metallic_factor: float,
                           roughness_factor: float, device, H: int, W: int):
        """Create a single mesh from merged layers."""
        
        # Merge layers by compositing them
        merged_rgb = np.zeros((H, W, 3), dtype=np.float32)
        merged_depth = np.zeros((H, W), dtype=np.float32) 
        merged_alpha = np.zeros((H, W), dtype=np.float32)
        
        # Sort layers by priority (background first)
        sorted_layers = sorted(layers, key=lambda x: x['priority'])
        
        for layer in sorted_layers:
            alpha = layer['alpha']
            rgb = layer['rgb']
            depth = layer['depth']
            
            # Alpha blend
            merged_rgb = merged_rgb * (1 - alpha[..., np.newaxis]) + rgb * alpha[..., np.newaxis]
            merged_depth = merged_depth * (1 - alpha) + depth * alpha
            merged_alpha = np.maximum(merged_alpha, alpha)
        
        log.info(f"Merged {len(layers)} layers, coverage: {merged_alpha.mean():.2%}")
        
        # Create mesh from merged data
        return self._create_single_mesh(merged_rgb, merged_depth, merged_alpha, mesh_resolution,
                                      depth_scale, remove_edge, smooth_normals, metallic_factor,
                                      roughness_factor, device, H, W)
    
    def _create_layer_mesh(self, layer: Dict, mesh_resolution: int, depth_scale: float,
                          remove_edge: bool, smooth_normals: bool, metallic_factor: float,
                          roughness_factor: float, device, H: int, W: int):
        """Create mesh from a single layer."""
        
        rgb = layer['rgb']
        depth = layer['depth'] 
        alpha = layer['alpha']
        
        log.info(f"Creating mesh for {layer['type']} layer, coverage: {alpha.mean():.2%}")
        
        return self._create_single_mesh(rgb, depth, alpha, mesh_resolution, depth_scale,
                                      remove_edge, smooth_normals, metallic_factor, 
                                      roughness_factor, device, H, W)
    
    def _create_legacy_inputs(self, layer_rgb: torch.Tensor, layer_depth: torch.Tensor, 
                            layer_alpha: torch.Tensor, mesh_resolution: int, depth_scale: float,
                            remove_edge: bool, smooth_normals: bool, metallic_factor: float, 
                            roughness_factor: float, device):
        """Create mesh from legacy single-layer inputs."""
        
        # Convert to numpy
        rgb_np = layer_rgb[0].cpu().numpy().astype(np.float32)  # [H, W, 3]
        depth_np = layer_depth[0, :, :, 0].cpu().numpy().astype(np.float32)  # [H, W]
        alpha_np = layer_alpha[0, :, :, 0].cpu().numpy().astype(np.float32)  # [H, W]
        
        H, W = depth_np.shape
        
        log.info(f"Creating spherical mesh from {W}x{H} legacy ERP data")
        
        return self._create_single_mesh(rgb_np, depth_np, alpha_np, mesh_resolution, depth_scale,
                                      remove_edge, smooth_normals, metallic_factor,
                                      roughness_factor, device, H, W)
    
    def _create_single_mesh(self, rgb_np: np.ndarray, depth_np: np.ndarray, alpha_np: np.ndarray,
                           mesh_resolution: int, depth_scale: float, remove_edge: bool, 
                           smooth_normals: bool, metallic_factor: float, roughness_factor: float,
                           device, H: int, W: int):
        
        # Resize to mesh resolution if needed
        if W != mesh_resolution or H != mesh_resolution // 2:
            target_h = mesh_resolution // 2
            target_w = mesh_resolution
            
            rgb_np = self._resize_array(rgb_np, (target_h, target_w))
            depth_np = self._resize_array(depth_np, (target_h, target_w))
            alpha_np = self._resize_array(alpha_np, (target_h, target_w))
            
            H, W = target_h, target_w
        
        # Apply depth scaling
        depth_np = depth_np * depth_scale
        
        # Create 3D points using spherical projection
        points_3d = self._erp_to_spherical_points(depth_np, W, H)
        
        # Create UV coordinates for texture mapping
        uv_coords = image_uv(width=W, height=H)
        
        # Create mask for valid regions
        valid_mask = alpha_np > 0.1
        if remove_edge:
            edge_mask = depth_edge(depth_np, mask=valid_mask, rtol=0.02)
            valid_mask = valid_mask & ~edge_mask
        
        # Generate mesh using spherical triangulation
        faces, vertices, vertex_colors, vertex_uvs = image_mesh(
            points_3d,
            rgb_np,
            uv_coords,
            mask=valid_mask,
            tri=True
        )
        
        # Flip coordinates for proper orientation
        vertices = vertices * [1, -1, -1]
        vertex_uvs = vertex_uvs * [1, -1] + [0, 1]
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=trimesh.visual.texture.TextureVisuals(
                uv=vertex_uvs,
                material=trimesh.visual.material.PBRMaterial(
                    baseColorTexture=Image.fromarray((rgb_np * 255).astype(np.uint8)),
                    metallicFactor=metallic_factor,
                    roughnessFactor=roughness_factor
                )
            ),
            process=False
        )
        
        # Compute normals if requested
        if smooth_normals:
            mesh.vertex_normals
        
        # Create visualization outputs
        depth_vis = self._colorize_depth(depth_np, valid_mask)
        depth_tensor = torch.from_numpy(depth_vis).unsqueeze(0).to(device)
        
        normals_vis = self._compute_and_visualize_normals(points_3d, valid_mask)
        normals_tensor = torch.from_numpy(normals_vis).unsqueeze(0).to(device)
        
        log.info(f"Created spherical mesh with {len(vertices)} vertices, {len(faces)} faces")
        
        return (mesh, depth_tensor, normals_tensor)
    
    def _resize_array(self, arr: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        """Resize array using OpenCV for better quality."""
        import cv2
        
        if len(arr.shape) == 2:
            return cv2.resize(arr, (new_size[1], new_size[0]), interpolation=cv2.INTER_LANCZOS4)
        else:
            return cv2.resize(arr, (new_size[1], new_size[0]), interpolation=cv2.INTER_LANCZOS4)
    
    def _erp_to_spherical_points(self, depth: np.ndarray, width: int, height: int) -> np.ndarray:
        """Convert ERP depth map to 3D spherical points."""
        points = np.zeros((height, width, 3), dtype=np.float32)
        
        for v in range(height):
            for u in range(width):
                if depth[v, u] <= 0:
                    continue
                    
                # Get unit sphere direction
                x_unit, y_unit, z_unit = SphericalProjection.erp_to_xyz(u, v, width, height)
                
                # Scale by depth
                d = depth[v, u]
                points[v, u] = [x_unit * d, y_unit * d, z_unit * d]
        
        return points
    
    def _colorize_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Colorize depth map for visualization."""
        colored = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
        
        if mask.sum() > 0:
            valid_depth = depth[mask]
            min_d, max_d = valid_depth.min(), valid_depth.max()
            
            if max_d > min_d:
                normalized = (depth - min_d) / (max_d - min_d)
                normalized = np.clip(normalized, 0, 1)
                
                # Apply colormap (simple grayscale to RGB)
                colored[:, :, 0] = normalized
                colored[:, :, 1] = normalized  
                colored[:, :, 2] = normalized
                
                # Apply mask
                colored = colored * mask[..., np.newaxis]
        
        return colored
    
    def _compute_and_visualize_normals(self, points: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Compute and visualize surface normals."""
        H, W = points.shape[:2]
        normals = np.zeros_like(points)
        
        # Compute normals using gradient method
        for v in range(1, H-1):
            for u in range(1, W-1):
                if not mask[v, u]:
                    continue
                
                # Get neighboring points
                p_center = points[v, u]
                p_right = points[v, (u+1) % W]  # Handle wrap-around
                p_down = points[v+1, u]
                p_left = points[v, (u-1) % W]   # Handle wrap-around  
                p_up = points[v-1, u]
                
                # Compute gradients
                grad_u = (p_right - p_left) / 2
                grad_v = (p_down - p_up) / 2
                
                # Cross product gives normal
                normal = np.cross(grad_u, grad_v)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                
                normals[v, u] = normal
        
        # Convert to RGB visualization (map [-1,1] to [0,1])
        normals_vis = (normals + 1) / 2
        normals_vis = normals_vis * mask[..., np.newaxis]
        
        return normals_vis.astype(np.float32)


class Spherical_Mesh_Exporter:
    """Export spherical meshes to various formats."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "save_format": (["glb", "ply", "obj"], {"default": "glb"}),
                "filename_prefix": ("STRING", {"default": "3D/Spherical"}),
                "include_texture": ("BOOLEAN", {"default": True}),
                "optimize_mesh": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export_mesh"
    CATEGORY = "MoGe360/Meshing"
    OUTPUT_NODE = True
    DESCRIPTION = "Export spherical mesh to file"

    def export_mesh(self, mesh: trimesh.Trimesh, save_format: str, filename_prefix: str,
                   include_texture: bool, optimize_mesh: bool):
        
        if mesh is None:
            return ("",)
        
        # Get output path
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        
        # Optimize mesh if requested
        if optimize_mesh:
            # Remove duplicate vertices
            mesh.remove_duplicate_faces()
            mesh.remove_degenerate_faces()
            
            # Merge close vertices
            mesh.merge_vertices()
        
        # Export based on format
        extension = save_format.lower()
        output_path = Path(full_output_folder) / f'{filename}_{counter:05}_.{extension}'
        output_path.parent.mkdir(exist_ok=True)
        
        try:
            if save_format == 'glb':
                mesh.export(output_path)
            elif save_format == 'ply':
                # For PLY, extract vertex colors if no texture
                if not include_texture and hasattr(mesh.visual, 'vertex_colors'):
                    ply_mesh = trimesh.Trimesh(
                        vertices=mesh.vertices,
                        faces=mesh.faces,
                        vertex_colors=mesh.visual.vertex_colors,
                        process=False
                    )
                    ply_mesh.export(output_path)
                else:
                    mesh.export(output_path)
            elif save_format == 'obj':
                mesh.export(output_path)
            
            relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{extension}'
            
            log.info(f"Exported spherical mesh to: {output_path}")
            
            return (str(relative_path),)
            
        except Exception as e:
            log.error(f"Failed to export mesh: {e}")
            return ("",)