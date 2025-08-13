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
                "layer_stack": ("LAYER_STACK", {
                    "tooltip": "Aligned layer stack from Depth_Align_Layers containing layers with consistent depth and refined alpha masks. Each layer will be converted to 3D mesh geometry."
                }),
                "mesh_resolution": ("INT", {
                    "default": 1024, "min": 512, "max": 2048, "step": 128,
                    "tooltip": "Target resolution for mesh generation. Higher values = more detail but larger files and slower processing. 1024 is good balance for most scenes."
                }),
                "layer_selection": (["all", "sky_only", "background_only", "objects_only"], {
                    "default": "all",
                    "tooltip": "Which layers to include in mesh generation: 'all' creates from all layers, other options filter to specific layer types for targeted mesh creation."
                }),
                "depth_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "Global depth scaling factor applied to all layers. 1.0 = original depth scale, >1.0 = exaggerated depth, <1.0 = flattened depth. Affects 3D parallax."
                }),
                "remove_edge": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove mesh faces at depth discontinuities to prevent stretched triangles. Recommended for clean geometry but may create holes at object boundaries."
                }),
                "smooth_normals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Compute smooth vertex normals for better lighting. Creates softer appearance but may blur sharp edges. Disable for faceted/angular look."
                }),
                "metallic_factor": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "PBR metallic factor for material properties. 0.0 = dielectric (non-metal), 1.0 = fully metallic. Affects how light reflects off the mesh surface."
                }),
                "roughness_factor": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "PBR roughness factor for material properties. 0.0 = mirror-like, 1.0 = completely rough/matte. Controls surface reflectivity and highlights."
                }),
                "merge_layers": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Composite all selected layers into single mesh. True = one combined mesh, False = separate mesh per layer. Merging simplifies output but loses layer separation."
                }),
            },
            "optional": {
                "layer_rgb": ("IMAGE", {
                    "tooltip": "Legacy single layer RGB input. Only used if layer_stack is not provided. For backward compatibility with older workflows."
                }),
                "layer_depth": ("IMAGE", {
                    "tooltip": "Legacy single layer depth input. Must be provided with layer_rgb and layer_alpha for legacy mode operation."
                }),
                "layer_alpha": ("IMAGE", {
                    "tooltip": "Legacy single layer alpha mask input. Defines valid regions for mesh generation in legacy mode."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE")
    RETURN_NAMES = ("spherical_mesh", "sphere_depth", "sphere_normals")
    OUTPUT_TOOLTIPS = (
        "Generated 3D mesh with proper spherical geometry, UV mapping, and PBR materials. Ready for export or further processing in 3D applications.",
        "Visualization of depth map used for mesh generation with color coding. Useful for verifying depth quality and identifying potential mesh issues.",
        "Visualization of computed surface normals as RGB image. Helps verify mesh quality and normal smoothness for proper lighting calculations."
    )
    FUNCTION = "create_spherical_mesh"
    CATEGORY = "MoGe360/Meshing"
    DESCRIPTION = "Convert aligned layer stack to 3D meshes using proper spherical projection. Creates textured meshes with PBR materials and correct UV mapping for 360° content."

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
        # Handle both 'dimensions' and 'image_dimensions' keys for compatibility
        if 'dimensions' in layer_stack:
            H, W = layer_stack['dimensions']
        elif 'image_dimensions' in layer_stack:
            H, W = layer_stack['image_dimensions']
        else:
            # Fallback - get dimensions from first layer
            first_layer = layers[0]
            if 'rgb' in first_layer:
                H, W = first_layer['rgb'].shape[:2]
            else:
                raise ValueError("Cannot determine layer stack dimensions")
        
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
        """Create a single mesh from merged layers with per-layer depth processing."""
        
        # Process each layer individually to preserve depth detail
        processed_layers = []
        
        log.info(f"Processing {len(layers)} layers individually for depth preservation")
        
        # Sort layers by priority (background first)
        sorted_layers = sorted(layers, key=lambda x: x['priority'])
        
        for i, layer in enumerate(sorted_layers):
            layer_type = layer['type']
            alpha = layer['alpha']
            rgb = layer['rgb']
            depth = layer['depth']
            
            # Apply per-layer depth normalization based on layer type
            processed_depth = self._normalize_layer_depth(depth, alpha, layer_type, depth_scale)
            
            processed_layers.append({
                'rgb': rgb,
                'depth': processed_depth, 
                'alpha': alpha,
                'type': layer_type
            })
            
            # Log depth statistics
            if alpha.sum() > 0:
                valid_depths = processed_depth[alpha > 0.1]
                if len(valid_depths) > 0:
                    log.info(f"  Layer {i} ({layer_type}): depth range [{valid_depths.min():.3f}, {valid_depths.max():.3f}], variation: {valid_depths.max() - valid_depths.min():.3f}")
        
        # Now merge the processed layers
        merged_rgb = np.zeros((H, W, 3), dtype=np.float32)
        merged_depth = np.zeros((H, W), dtype=np.float32) 
        merged_alpha = np.zeros((H, W), dtype=np.float32)
        
        for layer in processed_layers:
            alpha = layer['alpha']
            rgb = layer['rgb']
            depth = layer['depth']
            
            # Alpha blend
            merged_rgb = merged_rgb * (1 - alpha[..., np.newaxis]) + rgb * alpha[..., np.newaxis]
            merged_depth = merged_depth * (1 - alpha) + depth * alpha
            merged_alpha = np.maximum(merged_alpha, alpha)
        
        log.info(f"Merged {len(layers)} layers, coverage: {merged_alpha.mean():.2%}")
        
        # Create mesh from merged data - skip additional depth normalization since we did it per-layer
        return self._create_single_mesh_prenormalized(merged_rgb, merged_depth, merged_alpha, mesh_resolution,
                                                    remove_edge, smooth_normals, metallic_factor,
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
        
        # Create mask for valid regions with adaptive threshold
        alpha_threshold = 0.01  # More lenient threshold
        valid_mask = alpha_np > alpha_threshold
        
        # Log coverage information
        initial_coverage = valid_mask.mean()
        log.info(f"Initial alpha mask coverage: {initial_coverage:.2%} with threshold {alpha_threshold}")
        
        # If coverage is very low, try even more lenient threshold
        if initial_coverage < 0.05:  # Less than 5%
            alpha_threshold = 0.001
            valid_mask = alpha_np > alpha_threshold
            revised_coverage = valid_mask.mean()
            log.info(f"Revised alpha mask coverage: {revised_coverage:.2%} with threshold {alpha_threshold}")
        
        if remove_edge:
            edge_mask = depth_edge(depth_np, mask=valid_mask, rtol=0.02)
            valid_mask = valid_mask & ~edge_mask
            final_coverage = valid_mask.mean()
            log.info(f"Final mask coverage after edge removal: {final_coverage:.2%}")
        
        # Generate mesh using spherical triangulation
        faces, vertices, vertex_colors, vertex_uvs = image_mesh(
            points_3d,
            rgb_np,
            uv_coords,
            mask=valid_mask,
            tri=True
        )
        
        # Fix coordinate system for proper orientation (mountains right-side up)
        # Standard ERP has Y pointing up, Z pointing forward, X pointing right
        # We need to ensure mountains appear above the horizon, not below
        vertices = vertices * [1, 1, -1]  # Keep Y positive for mountains above horizon
        vertex_uvs = vertex_uvs * [1, -1] + [0, 1]  # Flip V coordinate for texture mapping
        
        log.info(f"Vertex coordinate ranges: X=[{vertices[:,0].min():.2f}, {vertices[:,0].max():.2f}], "
                f"Y=[{vertices[:,1].min():.2f}, {vertices[:,1].max():.2f}], "
                f"Z=[{vertices[:,2].min():.2f}, {vertices[:,2].max():.2f}]")
        
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
        """Convert ERP depth map to 3D spherical points using vectorized operations."""
        
        # Create coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        
        # Normalize to [0, 1]
        lon_norm = u_coords.astype(np.float32) / width
        lat_norm = v_coords.astype(np.float32) / height
        
        # Convert to spherical coordinates
        longitude = (lon_norm - 0.5) * 2 * np.pi  # -π to π
        latitude = (0.5 - lat_norm) * np.pi       # -π/2 to π/2
        
        # Convert to unit sphere Cartesian coordinates
        x_unit = np.cos(latitude) * np.cos(longitude)
        y_unit = np.sin(latitude)
        z_unit = np.cos(latitude) * np.sin(longitude)
        
        # Scale by depth values
        # Ensure depth has proper range - normalize if needed
        depth_processed = depth.copy()
        
        # Apply depth scaling and handle invalid values
        valid_mask = depth_processed > 0
        if valid_mask.sum() > 0:
            valid_depths = depth_processed[valid_mask]
            depth_min, depth_max = valid_depths.min(), valid_depths.max()
            
            # Intelligent depth normalization focusing on foreground detail
            # Use histogram analysis to separate sky from terrain
            depth_hist, depth_bins = np.histogram(valid_depths, bins=50)
            
            # Find the largest gap in the histogram (likely between terrain and sky)
            bin_centers = (depth_bins[:-1] + depth_bins[1:]) / 2
            
            # Look for bimodal distribution
            peak_indices = []
            for i in range(1, len(depth_hist) - 1):
                if depth_hist[i] > depth_hist[i-1] and depth_hist[i] > depth_hist[i+1]:
                    peak_indices.append(i)
            
            # If we have clear peaks, use the gap between them
            if len(peak_indices) >= 2:
                # Find the largest gap between peaks
                gap_sizes = []
                gap_positions = []
                for i in range(len(peak_indices) - 1):
                    peak1_pos = bin_centers[peak_indices[i]]
                    peak2_pos = bin_centers[peak_indices[i+1]]
                    gap_size = peak2_pos - peak1_pos
                    gap_sizes.append(gap_size)
                    gap_positions.append((peak1_pos + peak2_pos) / 2)
                
                if gap_sizes:
                    largest_gap_idx = np.argmax(gap_sizes)
                    sky_threshold = gap_positions[largest_gap_idx]
                else:
                    # Fallback: use 85th percentile
                    sky_threshold = np.percentile(valid_depths, 85)
            else:
                # Fallback: use 85th percentile  
                sky_threshold = np.percentile(valid_depths, 85)
            
            # Ensure we have enough foreground data
            foreground_mask = valid_depths < sky_threshold
            fg_ratio = foreground_mask.sum() / len(valid_depths)
            
            if fg_ratio > 0.3:  # At least 30% foreground
                fg_depths = valid_depths[foreground_mask]
                fg_min, fg_max = fg_depths.min(), fg_depths.max()
                
                if fg_max > fg_min:
                    # Give mountains much more detail: map to 1-9, sky to 9-10
                    depth_processed_new = depth_processed.copy()
                    fg_global_mask = (depth_processed > 0) & (depth_processed < sky_threshold)
                    sky_global_mask = (depth_processed > 0) & (depth_processed >= sky_threshold)
                    
                    # Normalize foreground with maximum detail (80% of range)
                    depth_processed_new[fg_global_mask] = 1 + 8 * (depth_processed[fg_global_mask] - fg_min) / (fg_max - fg_min)
                    # Place sky at far distance (20% of range)
                    depth_processed_new[sky_global_mask] = 9 + 1 * (depth_processed[sky_global_mask] - sky_threshold) / (depth_max - sky_threshold + 1e-6)
                    
                    depth_processed = depth_processed_new
                    log.info(f"Smart depth normalization: foreground [{fg_min:.3f}, {fg_max:.3f}] -> [1, 9] (fg ratio: {fg_ratio:.1%}), sky [{sky_threshold:.3f}, {depth_max:.3f}] -> [9, 10]")
                else:
                    # Fallback to full range normalization
                    log.info(f"Normalizing full depth range from [{depth_min:.3f}, {depth_max:.3f}] to [1, 10]")
                    depth_processed[valid_mask] = 1 + 9 * (valid_depths - depth_min) / (depth_max - depth_min)
            else:
                # Fallback to full range normalization
                log.info(f"Normalizing full depth range from [{depth_min:.3f}, {depth_max:.3f}] to [1, 10]")
                depth_processed[valid_mask] = 1 + 9 * (valid_depths - depth_min) / (depth_max - depth_min)
            
            # Apply global scaling to make mesh more reasonable size
            depth_processed = depth_processed * 5.0  # Scale factor for better viewing
        
        # Create 3D points
        points = np.zeros((height, width, 3), dtype=np.float32)
        points[:, :, 0] = x_unit * depth_processed
        points[:, :, 1] = y_unit * depth_processed  
        points[:, :, 2] = z_unit * depth_processed
        
        # Set invalid points to zero
        invalid_mask = depth <= 0
        points[invalid_mask] = 0
        
        log.info(f"Generated spherical points: range X=[{points[:,:,0].min():.2f}, {points[:,:,0].max():.2f}], "
                f"Y=[{points[:,:,1].min():.2f}, {points[:,:,1].max():.2f}], "
                f"Z=[{points[:,:,2].min():.2f}, {points[:,:,2].max():.2f}]")
        
        return points
    
    def _colorize_depth(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Colorize depth map for visualization."""
        colored = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.float32)
        
        # Handle different cases for depth normalization
        if mask.sum() > 0:
            valid_depth = depth[mask]
            if len(valid_depth) > 0:
                min_d, max_d = valid_depth.min(), valid_depth.max()
                
                if max_d > min_d:
                    normalized = (depth - min_d) / (max_d - min_d)
                else:
                    # If all depths are the same, normalize by absolute value
                    if max_d > 0:
                        normalized = depth / max_d
                    else:
                        normalized = np.ones_like(depth) * 0.5
            else:
                normalized = np.ones_like(depth) * 0.5
        else:
            # If no valid mask, try to normalize the entire depth map
            if depth.max() > 0:
                normalized = depth / depth.max()
            else:
                normalized = np.ones_like(depth) * 0.5
        
        normalized = np.clip(normalized, 0, 1)
        
        # Create a colormap (jet-like for better visualization)
        # Blue (near) -> Green -> Yellow -> Red (far)
        colored[:, :, 0] = np.maximum(0, np.minimum(1, 4 * normalized - 2))  # Red channel
        colored[:, :, 1] = np.maximum(0, 1 - 2 * np.abs(normalized - 0.5))   # Green channel  
        colored[:, :, 2] = np.maximum(0, np.minimum(1, 2 - 4 * normalized))  # Blue channel
        
        # Apply mask if provided
        if mask.sum() > 0:
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
    
    def _normalize_layer_depth(self, depth: np.ndarray, alpha: np.ndarray, layer_type: str, depth_scale: float) -> np.ndarray:
        """Apply per-layer depth normalization to preserve detail within each layer."""
        
        # Get valid depth values for this layer
        valid_mask = (alpha > 0.1) & (depth > 0)
        
        if valid_mask.sum() == 0:
            return depth * depth_scale
        
        valid_depths = depth[valid_mask]
        depth_min, depth_max = valid_depths.min(), valid_depths.max()
        
        if layer_type == 'sky':
            # Sky: keep at far distance, no normalization needed
            # Sky should already be at ~1000 units from layer creation
            processed_depth = depth.copy()
            
        elif layer_type == 'background':
            # Background (mountains): apply smart normalization for maximum terrain detail
            processed_depth = depth.copy()
            
            if depth_max > depth_min:
                # Apply the same smart normalization as in single mesh creation
                # but only to this layer's depth values to preserve mountain detail
                normalized_depths = self._apply_smart_terrain_normalization(valid_depths)
                
                # Map back to the depth array
                processed_depth[valid_mask] = normalized_depths
                
                # Scale by user preference
                processed_depth = processed_depth * depth_scale
                
                log.info(f"Background layer depth normalized: [{depth_min:.3f}, {depth_max:.3f}] -> [{processed_depth[valid_mask].min():.3f}, {processed_depth[valid_mask].max():.3f}]")
            else:
                processed_depth = processed_depth * depth_scale
                
        else:  # 'object' layers
            # Objects: normalize but keep closer than background
            processed_depth = depth.copy()
            
            if depth_max > depth_min:
                # Normalize to 0.5-8 range (closer than background)
                normalized = 0.5 + 7.5 * (valid_depths - depth_min) / (depth_max - depth_min)
                processed_depth[valid_mask] = normalized
                processed_depth = processed_depth * depth_scale
                
                log.info(f"Object layer depth normalized: [{depth_min:.3f}, {depth_max:.3f}] -> [{processed_depth[valid_mask].min():.3f}, {processed_depth[valid_mask].max():.3f}]")
            else:
                processed_depth = processed_depth * depth_scale
        
        return processed_depth
    
    def _apply_smart_terrain_normalization(self, depths: np.ndarray) -> np.ndarray:
        """Apply smart normalization focusing on terrain detail - adapted from single mesh logic."""
        
        # For background/terrain layer, we want maximum detail
        # Normalize to 1-9 range to give full depth variation to terrain
        depth_min, depth_max = depths.min(), depths.max()
        
        if depth_max > depth_min:
            # Give terrain maximum depth range for detail
            normalized = 1 + 8 * (depths - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.full_like(depths, 5.0)  # Middle value if no variation
        
        return normalized
    
    def _create_single_mesh_prenormalized(self, rgb_np: np.ndarray, depth_np: np.ndarray, alpha_np: np.ndarray,
                                        mesh_resolution: int, remove_edge: bool, smooth_normals: bool, 
                                        metallic_factor: float, roughness_factor: float, device, H: int, W: int):
        """Create mesh from pre-normalized depth data (skips depth normalization)."""
        
        # Resize to mesh resolution if needed
        if W != mesh_resolution or H != mesh_resolution // 2:
            target_h = mesh_resolution // 2
            target_w = mesh_resolution
            
            rgb_np = self._resize_array(rgb_np, (target_h, target_w))
            depth_np = self._resize_array(depth_np, (target_h, target_w))
            alpha_np = self._resize_array(alpha_np, (target_h, target_w))
            
            H, W = target_h, target_w
        
        # Create 3D points using spherical projection - skip depth normalization
        points_3d = self._erp_to_spherical_points_prenormalized(depth_np, W, H)
        
        # Create UV coordinates for texture mapping
        uv_coords = image_uv(width=W, height=H)
        
        # Create mask for valid regions
        alpha_threshold = 0.01
        valid_mask = alpha_np > alpha_threshold
        
        # Log coverage information
        coverage = valid_mask.mean()
        log.info(f"Pre-normalized mesh coverage: {coverage:.2%}")
        
        if remove_edge:
            edge_mask = depth_edge(depth_np, mask=valid_mask, rtol=0.02)
            valid_mask = valid_mask & ~edge_mask
            final_coverage = valid_mask.mean()
            log.info(f"Final mask coverage after edge removal: {final_coverage:.2%}")
        
        # Generate mesh using spherical triangulation
        faces, vertices, vertex_colors, vertex_uvs = image_mesh(
            points_3d,
            rgb_np,
            uv_coords,
            mask=valid_mask,
            tri=True
        )
        
        # Fix coordinate system for proper orientation
        vertices = vertices * [1, 1, -1]
        vertex_uvs = vertex_uvs * [1, -1] + [0, 1]
        
        log.info(f"Pre-normalized mesh vertex ranges: X=[{vertices[:,0].min():.2f}, {vertices[:,0].max():.2f}], "
                f"Y=[{vertices[:,1].min():.2f}, {vertices[:,1].max():.2f}], "
                f"Z=[{vertices[:,2].min():.2f}, {vertices[:,2].max():.2f}]")
        
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
        
        log.info(f"Created pre-normalized spherical mesh with {len(vertices)} vertices, {len(faces)} faces")
        
        return (mesh, depth_tensor, normals_tensor)
    
    def _erp_to_spherical_points_prenormalized(self, depth: np.ndarray, width: int, height: int) -> np.ndarray:
        """Convert ERP depth map to 3D spherical points - assumes depth is already normalized."""
        
        # Create coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        
        # Normalize to [0, 1]
        lon_norm = u_coords.astype(np.float32) / width
        lat_norm = v_coords.astype(np.float32) / height
        
        # Convert to spherical coordinates
        longitude = (lon_norm - 0.5) * 2 * np.pi  # -π to π
        latitude = (0.5 - lat_norm) * np.pi       # -π/2 to π/2
        
        # Convert to unit sphere Cartesian coordinates
        x_unit = np.cos(latitude) * np.cos(longitude)
        y_unit = np.sin(latitude)
        z_unit = np.cos(latitude) * np.sin(longitude)
        
        # Use depth directly since it's already normalized per-layer
        depth_processed = depth.copy()
        
        # Apply global scaling to make mesh reasonable size
        depth_processed = depth_processed * 5.0  # Scale factor for better viewing
        
        # Create 3D points
        points = np.zeros((height, width, 3), dtype=np.float32)
        points[:, :, 0] = x_unit * depth_processed
        points[:, :, 1] = y_unit * depth_processed  
        points[:, :, 2] = z_unit * depth_processed
        
        # Set invalid points to zero
        invalid_mask = depth <= 0
        points[invalid_mask] = 0
        
        log.info(f"Generated pre-normalized spherical points: range X=[{points[:,:,0].min():.2f}, {points[:,:,0].max():.2f}], "
                f"Y=[{points[:,:,1].min():.2f}, {points[:,:,1].max():.2f}], "
                f"Z=[{points[:,:,2].min():.2f}, {points[:,:,2].max():.2f}]")
        
        return points


class Spherical_Mesh_Exporter:
    """Export spherical meshes to various formats."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH", {
                    "tooltip": "3D mesh from Layer_Mesher_Spherical to export to file. Must be a valid Trimesh object with geometry and optional textures."
                }),
                "save_format": (["glb", "ply", "obj"], {
                    "default": "glb",
                    "tooltip": "Export file format: 'glb' is modern binary format with full PBR support, 'ply' is simple for point clouds/vertex colors, 'obj' is widely compatible."
                }),
                "filename_prefix": ("STRING", {
                    "default": "3D/Spherical",
                    "tooltip": "Filename prefix for exported mesh. Can include subdirectory path. Final filename will include timestamp and format extension automatically."
                }),
                "include_texture": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include texture/material information in export. True = full PBR materials, False = geometry only or vertex colors. Some formats always include textures."
                }),
                "optimize_mesh": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply mesh optimization before export: removes duplicate vertices/faces, merges close vertices. Reduces file size but may slightly alter geometry."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_TOOLTIPS = (
        "Relative path to exported mesh file. Can be used to locate the saved 3D model or passed to other tools for further processing.",
    )
    FUNCTION = "export_mesh"
    CATEGORY = "MoGe360/Meshing"
    OUTPUT_NODE = True
    DESCRIPTION = "Export 3D mesh to file in specified format with optimization options. Supports GLB, PLY, and OBJ formats with proper material and texture handling."

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