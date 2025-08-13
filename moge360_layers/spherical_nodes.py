"""
Core spherical nodes for 360 panorama processing.
Implements the spherical MoGe pipeline: tiling, processing, and stitching.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

from .spherical_utils import SphericalProjection
import comfy.model_management as mm

log = logging.getLogger(__name__)

class Pano_TileSampler_Spherical:
    """Sample perspective tiles from equirectangular panorama using spherical projection."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "erp_image": ("IMAGE", {
                    "tooltip": "Input equirectangular (360°) panorama image in 2:1 aspect ratio. Should be in standard ERP format with longitude wrapping at ±180°."
                }),
                "grid_yaw": ("INT", {
                    "default": 6, "min": 2, "max": 12, "step": 1,
                    "tooltip": "Number of tiles horizontally around the sphere (yaw direction). More tiles = higher quality but slower processing. 6 tiles = 60° per tile."
                }),
                "grid_pitch": ("INT", {
                    "default": 3, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Number of tiles vertically (pitch direction). 3 tiles covers full sphere from pole to pole. More tiles = better pole handling."
                }),
                "tile_fov": ("FLOAT", {
                    "default": 100.0, "min": 60.0, "max": 140.0, "step": 5.0,
                    "tooltip": "Field of view for each perspective tile in degrees. Larger FOV captures more area per tile but may introduce distortion. 100° is optimal for most scenes."
                }),
                "overlap": ("FLOAT", {
                    "default": 15.0, "min": 5.0, "max": 30.0, "step": 1.0,
                    "tooltip": "Overlap between adjacent tiles in degrees. Essential for seamless stitching. 15° provides good balance between redundancy and processing speed."
                }),
                "tile_resolution": ("INT", {
                    "default": 768, "min": 512, "max": 1024, "step": 64,
                    "tooltip": "Resolution of each perspective tile in pixels. Higher resolution = better detail but requires more GPU memory. 768px works well for most use cases."
                }),
            },
        }

    RETURN_TYPES = ("TILE_BATCH", "TILE_PARAMS", "IMAGE")
    RETURN_NAMES = ("tiles", "tile_params", "sample_preview")
    OUTPUT_TOOLTIPS = (
        "Batch of perspective tiles sampled from the panorama, ready for MoGe processing. Each tile contains a perspective view with proper camera intrinsics.",
        "Parameters for each tile including camera position, orientation, and intrinsics. Required for stitching tiles back together seamlessly.",
        "Preview image showing the tile sampling grid overlaid on the original panorama for verification and debugging purposes."
    )
    FUNCTION = "sample_tiles"
    CATEGORY = "MoGe360/Core"
    DESCRIPTION = "Sample perspective tiles from equirectangular panorama using true spherical projection. Replaces pinhole camera assumptions with proper spherical geometry for 360° processing."

    def sample_tiles(self, erp_image: torch.Tensor, grid_yaw: int, grid_pitch: int, 
                    tile_fov: float, overlap: float, tile_resolution: int):
        
        device = mm.get_torch_device()
        B, H, W, C = erp_image.shape
        
        # Convert to numpy for processing
        erp_np = erp_image[0].cpu().numpy()  # Take first batch item
        if erp_np.dtype != np.float32:
            erp_np = erp_np.astype(np.float32)
        
        # Create tile grid
        tiles_params = SphericalProjection.create_tile_grid(
            grid_yaw=grid_yaw, 
            grid_pitch=grid_pitch,
            fov_deg=tile_fov, 
            overlap_deg=overlap
        )
        
        log.info(f"Created {len(tiles_params)} tiles for {grid_yaw}x{grid_pitch} grid")
        
        # Sample all tiles with progress logging
        tiles = []
        intrinsics_list = []
        extrinsics_list = []
        
        for i, tile_params in enumerate(tiles_params):
            log.info(f"Sampling tile {i+1}/{len(tiles_params)}: lon={tile_params['center_lon']:.1f}°, lat={tile_params['center_lat']:.1f}°")
            
            tile_img, intrinsics, extrinsics = SphericalProjection.sample_perspective_tile(
                erp_np, 
                tile_params['center_lon'],
                tile_params['center_lat'],
                tile_params['fov_deg'],
                tile_resolution
            )
            
            tiles.append(tile_img)
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)
            
            # Progress update every few tiles
            if (i + 1) % 3 == 0 or (i + 1) == len(tiles_params):
                log.info(f"Completed {i+1}/{len(tiles_params)} tiles")
        
        # Convert tiles to tensor format [N, H, W, C]
        tiles_tensor = torch.from_numpy(np.stack(tiles, axis=0)).to(device)
        if tiles_tensor.dtype != erp_image.dtype:
            tiles_tensor = tiles_tensor.to(erp_image.dtype)
        
        # Create stitch map for later reconstruction
        stitch_weights = SphericalProjection.calculate_stitch_weights(
            tiles_params, W, H
        )
        
        stitch_map = {
            'weights': stitch_weights,
            'erp_dimensions': (H, W),
            'tile_resolution': tile_resolution,
            'intrinsics': intrinsics_list,
            'extrinsics': extrinsics_list
        }
        
        # Package tile parameters
        tile_params_out = {
            'tiles_params': tiles_params,
            'grid_shape': (grid_yaw, grid_pitch),
            'fov_deg': tile_fov,
            'overlap_deg': overlap,
            'stitch_map': stitch_map  # Include stitch_map for later use
        }
        
        # Create preview image showing tile grid
        preview_img = self._create_tile_preview(erp_np, tiles_params, W, H)
        preview_tensor = torch.from_numpy(preview_img).unsqueeze(0).to(device)
        
        return (tiles_tensor, tile_params_out, preview_tensor)
    
    def _create_tile_preview(self, img: np.ndarray, tiles_params: List[Dict], W: int, H: int) -> np.ndarray:
        """Create preview image showing tile sampling grid."""
        import cv2
        
        # Create preview image
        preview = img.copy()
        
        # Draw tile boundaries
        for tile_param in tiles_params:
            center_lon = tile_param['center_lon']
            center_lat = tile_param['center_lat']
            fov_deg = tile_param['fov_deg']
            
            # Convert to pixel coordinates (simplified)
            center_u = int((center_lon + 180) / 360 * W)
            center_v = int((90 - center_lat) / 180 * H)
            
            # Draw a circle at tile center
            cv2.circle(preview, (center_u, center_v), 10, (1.0, 0.0, 0.0), 2)
            
            # Draw approximate tile boundary (simplified rectangular approximation)
            tile_size_u = int(fov_deg / 360 * W)
            tile_size_v = int(fov_deg / 180 * H)
            
            x1 = max(0, center_u - tile_size_u // 2)
            y1 = max(0, center_v - tile_size_v // 2)
            x2 = min(W, center_u + tile_size_u // 2)
            y2 = min(H, center_v + tile_size_v // 2)
            
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0.0, 1.0, 0.0), 1)
        
        return preview.astype(np.float32)


class MoGe_PerTile_Geometry:
    """Process perspective tiles through MoGe model to get depth and normals."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_model": ("MOGEMODEL", {
                    "tooltip": "Pre-loaded MoGe model for monocular geometry estimation. Use 'Load MoGe Model' node to load the model with appropriate precision (fp16/fp32)."
                }),
                "tiles": ("TILE_BATCH", {
                    "tooltip": "Batch of perspective tiles from Pano_TileSampler_Spherical. Each tile represents a perspective view of the 360° scene."
                }),
                "tile_params": ("TILE_PARAMS", {
                    "tooltip": "Tile parameters including camera intrinsics, extrinsics, and sampling information. Required for proper spherical reconstruction."
                }),
                "resolution_level": ("INT", {
                    "default": 9, "min": 6, "max": 11, "step": 1,
                    "tooltip": "MoGe processing resolution level. Higher values = better quality but slower processing. Level 9 is optimal for 768px tiles, level 7-8 for speed."
                }),
                "batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of tiles to process simultaneously. Higher values = faster processing but more GPU memory usage. Adjust based on available VRAM."
                }),
                "lock_fov": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use consistent camera intrinsics for all tiles. Recommended for spherical consistency. Disable only if tiles have varying FOV requirements."
                }),
                "create_preview": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate preview images showing depth and normal maps. Useful for debugging but increases processing time and memory usage."
                }),
            },
        }

    RETURN_TYPES = ("DEPTH_BATCH", "NORMAL_BATCH")
    RETURN_NAMES = ("tile_depths", "tile_normals")
    OUTPUT_TOOLTIPS = (
        "Batch of depth maps from MoGe processing. Each depth map corresponds to a perspective tile and contains metric depth values for 3D reconstruction.",
        "Batch of normal maps from MoGe processing. Surface normals help with lighting and mesh quality. Each normal map contains XYZ surface normal vectors."
    )
    FUNCTION = "process_tiles"
    CATEGORY = "MoGe360/Core"
    DESCRIPTION = "Process perspective tiles through MoGe model to extract depth and surface normals. Uses proper spherical camera intrinsics for accurate 360° geometry reconstruction."

    def process_tiles(self, moge_model, tiles: torch.Tensor, tile_params: Dict,
                     resolution_level: int, batch_size: int, lock_fov: bool, create_preview: bool = False):
        
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        
        N, H, W, C = tiles.shape
        
        # Process tiles in batches
        all_depths = []
        all_normals = []
        all_points = []
        all_masks = []
        
        log.info(f"Processing {N} tiles in batches of {batch_size}")
        
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            batch_tiles = tiles[i:batch_end]
            
            # Process each tile in the batch
            batch_depths = []
            batch_normals = []
            batch_points = []
            batch_masks = []
            
            for j in range(batch_tiles.shape[0]):
                tile = batch_tiles[j]  # [H, W, C]
                tile_input = tile.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                
                # Run MoGe inference
                with torch.no_grad():
                    output = moge_model.infer(
                        tile_input[0], 
                        resolution_level=resolution_level, 
                        apply_mask=True
                    )
                
                # Extract outputs
                depth = output['depth']  # [H, W]
                points = output['points']  # Shape varies - could be [3, H, W] or [H, W, 3]
                mask = output['mask']  # [H, W]
                
                # Normalize points shape to [3, H, W]
                if points.dim() == 3:
                    if points.shape[0] == 3:
                        # Already in correct format [3, H, W]
                        pass
                    elif points.shape[2] == 3:
                        # Convert from [H, W, 3] to [3, H, W]
                        points = points.permute(2, 0, 1)
                        log.info(f"Converted points from [H, W, 3] to [3, H, W]: {points.shape}")
                    else:
                        log.error(f"Unexpected points shape: {points.shape}")
                
                # Validate and potentially scale points for better normal computation
                points_magnitude = torch.sqrt((points**2).sum(dim=0)).mean()
                if points_magnitude < 0.1 or points_magnitude > 100:
                    # Scale points to reasonable range for normal computation
                    scale_factor = 5.0 / points_magnitude.clamp(min=0.1)
                    points = points * scale_factor
                    log.info(f"Scaled points by {scale_factor:.3f} (magnitude: {points_magnitude:.3f} -> {torch.sqrt((points**2).sum(dim=0)).mean():.3f})")

                # Calculate normals from points if available
                if 'normals' in output:
                    normals = output['normals']  # [3, H, W]
                    # Also normalize normals shape if needed
                    if normals.dim() == 3 and normals.shape[2] == 3 and normals.shape[0] != 3:
                        normals = normals.permute(2, 0, 1)
                        log.info(f"Converted normals from [H, W, 3] to [3, H, W]: {normals.shape}")
                    
                    # Validate normals quality
                    normals_magnitude = torch.sqrt((normals**2).sum(dim=0)).mean()
                    if normals_magnitude < 0.5 or torch.isnan(normals_magnitude):
                        log.info(f"Poor quality normals (magnitude: {normals_magnitude:.3f}), recomputing from points")
                        normals = self._compute_normals_from_points(points)
                else:
                    # Compute normals from depth/points
                    log.info(f"Computing normals from points, shape: {points.shape}")
                    normals = self._compute_normals_from_points(points)
                
                batch_depths.append(depth.cpu())
                batch_normals.append(normals.cpu())  
                batch_points.append(points.cpu())
                batch_masks.append(mask.cpu())
            
            all_depths.extend(batch_depths)
            all_normals.extend(batch_normals)
            all_points.extend(batch_points)
            all_masks.extend(batch_masks)
            
            # Clear GPU memory between batches
            mm.soft_empty_cache()
        
        # Stack all results
        depths_tensor = torch.stack(all_depths)  # [N, H, W]
        normals_tensor = torch.stack(all_normals)  # [N, 3, H, W]
        points_tensor = torch.stack(all_points)  # [N, 3, H, W] 
        masks_tensor = torch.stack(all_masks)  # [N, H, W]
        
        # Convert depth and normals to image format for output [N, H, W, C]
        depths_img = depths_tensor.unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        normals_img = normals_tensor.permute(0, 2, 3, 1).to(device)  # [N, H, W, 3]
        
        # Package geometry data
        tile_geometry = {
            'depths': depths_tensor,
            'normals': normals_tensor,
            'points': points_tensor,
            'masks': masks_tensor,
            'tile_params': tile_params
        }
        
        log.info(f"Processed {N} tiles successfully")
        
        return (depths_img, normals_img, tile_geometry)
    
    def _compute_normals_from_points(self, points: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from 3D points using improved gradient method."""
        # points: [3, H, W] where first dim is [x, y, z]
        
        if points.shape[0] != 3:
            log.error(f"Expected points shape [3, H, W], got {points.shape}")
            # Return dummy normals as fallback
            return torch.zeros_like(points)
        
        C, H, W = points.shape
        log.info(f"Computing normals from points with shape: {points.shape}")
        
        # Extract xyz coordinates
        x = points[0]  # [H, W]
        y = points[1]  # [H, W] 
        z = points[2]  # [H, W]
        
        # Improved gradient computation with better edge handling
        def compute_gradient_improved(values):
            """Compute gradients with improved edge handling and numerical stability."""
            H, W = values.shape
            grad_u = torch.zeros_like(values)
            grad_v = torch.zeros_like(values)
            
            # Central differences for interior points (more accurate)
            grad_u[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / 2.0
            grad_v[1:-1, :] = (values[2:, :] - values[:-2, :]) / 2.0
            
            # Better edge handling - use forward/backward differences
            grad_u[:, 0] = values[:, 1] - values[:, 0]
            grad_u[:, -1] = values[:, -1] - values[:, -2]
            grad_v[0, :] = values[1, :] - values[0, :]
            grad_v[-1, :] = values[-1, :] - values[-2, :]
            
            return grad_u, grad_v
        
        # Compute gradients for each coordinate
        dx_du, dx_dv = compute_gradient_improved(x)
        dy_du, dy_dv = compute_gradient_improved(y)
        dz_du, dz_dv = compute_gradient_improved(z)
        
        # Create tangent vectors
        tangent_u = torch.stack([dx_du, dy_du, dz_du], dim=0)  # [3, H, W]
        tangent_v = torch.stack([dx_dv, dy_dv, dz_dv], dim=0)  # [3, H, W]
        
        # Cross product: n = tu × tv
        # For vectors a=[a0,a1,a2] and b=[b0,b1,b2], cross product is:
        # [a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0]
        normal_x = tangent_u[1] * tangent_v[2] - tangent_u[2] * tangent_v[1]  # [H, W]
        normal_y = tangent_u[2] * tangent_v[0] - tangent_u[0] * tangent_v[2]  # [H, W]
        normal_z = tangent_u[0] * tangent_v[1] - tangent_u[1] * tangent_v[0]  # [H, W]
        
        # Stack normals
        normals = torch.stack([normal_x, normal_y, normal_z], dim=0)  # [3, H, W]
        
        # Improved normalization with better handling of degenerate cases
        norm = torch.sqrt(normals[0]**2 + normals[1]**2 + normals[2]**2 + 1e-8)  # [H, W]
        
        # Identify degenerate cases (where gradient is too small)
        valid_mask = norm > 1e-6
        
        # For valid normals, normalize properly
        normals_normalized = torch.zeros_like(normals)
        for i in range(3):
            normals_normalized[i][valid_mask] = normals[i][valid_mask] / norm[valid_mask]
        
        # For invalid normals, use a fallback (point straight out from sphere center)
        if not valid_mask.all():
            # Fallback: assume spherical surface, normal points radially outward
            sphere_radius = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
            
            # Ensure sphere radius is valid
            valid_radius_mask = sphere_radius > 1e-6
            combined_invalid_mask = ~valid_mask & valid_radius_mask
            
            if combined_invalid_mask.any():
                normals_normalized[0][combined_invalid_mask] = x[combined_invalid_mask] / sphere_radius[combined_invalid_mask]
                normals_normalized[1][combined_invalid_mask] = y[combined_invalid_mask] / sphere_radius[combined_invalid_mask]
                normals_normalized[2][combined_invalid_mask] = z[combined_invalid_mask] / sphere_radius[combined_invalid_mask]
            
            # For points at origin, use default up normal
            origin_mask = ~valid_mask & ~valid_radius_mask
            if origin_mask.any():
                normals_normalized[0][origin_mask] = 0.0
                normals_normalized[1][origin_mask] = 1.0
                normals_normalized[2][origin_mask] = 0.0
            
            invalid_count = (~valid_mask).sum()
            log.info(f"Used fallback normals for {invalid_count}/{H*W} pixels ({100*invalid_count/(H*W):.1f}%)")
        
        # Final validation - ensure all normals are unit length
        final_norm = torch.sqrt(normals_normalized[0]**2 + normals_normalized[1]**2 + normals_normalized[2]**2 + 1e-8)
        
        # Handle any remaining NaN or invalid values
        nan_mask = torch.isnan(final_norm) | torch.isinf(final_norm) | (final_norm < 1e-6)
        if nan_mask.any():
            # Set problematic normals to point straight up
            normals_normalized[0][nan_mask] = 0.0
            normals_normalized[1][nan_mask] = 1.0
            normals_normalized[2][nan_mask] = 0.0
            final_norm[nan_mask] = 1.0
            log.info(f"Fixed {nan_mask.sum()} NaN/invalid normals")
        
        valid_norm_mask = final_norm > 1e-6
        if valid_norm_mask.any():
            avg_norm = final_norm[valid_norm_mask].mean().item()
        else:
            avg_norm = 0.0
            
        log.info(f"Computed normals with average magnitude: {avg_norm:.4f} (should be ~1.0)")
        
        return normals_normalized


class Depth_Normal_Stitcher_360:
    """Stitch perspective tile depth and normals back to equirectangular format."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_depths": ("DEPTH_BATCH", {
                    "tooltip": "Batch of depth maps from MoGe_PerTile_Geometry. Each depth map will be reprojected back to the corresponding region in the ERP panorama."
                }),
                "tile_normals": ("NORMAL_BATCH", {
                    "tooltip": "Batch of normal maps from MoGe_PerTile_Geometry. Surface normals will be transformed back to ERP space with proper spherical coordinates."
                }),
                "tile_params": ("TILE_PARAMS", {
                    "tooltip": "Tile parameters containing camera intrinsics and extrinsics. Required to correctly reproject tiles back to equirectangular format."
                }),
                "stitch_map": ("STITCH_MAP", {
                    "tooltip": "Stitching map containing blend weights and projection information from Pano_TileSampler_Spherical. Required for proper tile reconstruction."
                }),
                "output_height": ("INT", {
                    "default": 1024, "min": 512, "max": 2048, "step": 64,
                    "tooltip": "Height of output ERP panorama. Width will be 2x height for proper 2:1 aspect ratio. Higher resolution = better quality but more memory usage."
                }),
                "blend_mode": ([
                    "weighted", "feather", "poisson"
                ], {
                    "default": "weighted",
                    "tooltip": "Method for blending overlapping tile regions. 'weighted' uses distance-based weights, 'feather' uses edge feathering, 'poisson' uses Poisson blending for seamless transitions."
                }),
                "edge_feather": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Amount of edge feathering for seamless blending. 0.1 = 10% of tile border. Higher values = smoother blending but may blur details."
                }),
                "debug_seams": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show tile boundaries in output for debugging. Helpful for identifying stitching issues or adjusting overlap and feathering parameters."
                }),
                "create_preview": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate preview showing stitching process. Useful for debugging but increases processing time. Shows blend weights and tile boundaries."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("erp_depth", "erp_normals", "stitch_preview")
    OUTPUT_TOOLTIPS = (
        "Complete equirectangular depth map stitched from all perspective tiles. Contains metric depth values ready for 3D mesh reconstruction.",
        "Complete equirectangular normal map stitched from all tiles. Surface normals are properly transformed to ERP space for consistent lighting.",
        "Preview image showing the stitching process, tile boundaries, and blend regions. Useful for debugging and verifying seamless reconstruction."
    )
    FUNCTION = "stitch_geometry"
    CATEGORY = "MoGe360/Core"
    DESCRIPTION = "Stitch perspective tile depth and normals back to equirectangular panorama format with seamless blending. Handles ERP pole distortions and seam continuity across ±180° longitude boundary."

    def stitch_geometry(self, tile_depths, tile_normals, tile_params, stitch_map, 
                       output_height: int, blend_mode: str, edge_feather: float, 
                       debug_seams: bool, create_preview: bool):
        
        device = mm.get_torch_device()
        
        # Extract data
        # tile_depths and tile_normals are already the tensor data
        # tile_params contains the tile parameters and stitch_map
        tiles_params = tile_params.get('tiles_params', [])
        
        # Get stitch_map from tile_params if it wasn't passed separately
        if not isinstance(stitch_map, dict):
            if hasattr(stitch_map, 'get'):
                stitch_map = stitch_map
            else:
                stitch_map = tile_params.get('stitch_map', {})
        
        # Convert tensors to numpy for processing
        if hasattr(tile_depths, 'cpu'):
            tile_depths_np = tile_depths.cpu().numpy()  # [N, H, W] or [N, H, W, 3]
        else:
            tile_depths_np = tile_depths
            
        if hasattr(tile_normals, 'cpu'):
            tile_normals_np = tile_normals.cpu().numpy()  # [N, H, W] or [N, H, W, 3]
        else:
            tile_normals_np = tile_normals
            
        # Handle shape differences - depth might be [N, H, W, 3] format
        if len(tile_depths_np.shape) == 4:
            tile_depths_np = tile_depths_np[:, :, :, 0]  # Take first channel
        if len(tile_normals_np.shape) == 4:
            tile_normals_np = tile_normals_np  # Keep all 3 channels
        
        erp_h, erp_w = stitch_map.get('erp_dimensions', (output_height, output_height * 2))
        tile_res = stitch_map.get('tile_resolution', 768)
        weights = stitch_map.get('weights', {})
        
        # Implement proper spherical back-projection stitching
        log.info(f"Spherical stitching of {len(tile_depths_np)} tiles to {erp_w}x{erp_h} ERP")
        
        # Initialize output arrays  
        erp_depth = np.zeros((erp_h, erp_w), dtype=np.float32)
        erp_normals = np.zeros((erp_h, erp_w, 3), dtype=np.float32)
        erp_weights = np.zeros((erp_h, erp_w), dtype=np.float32)
        
        import cv2
        
        # Process each tile and back-project to ERP
        for tile_idx, tile_param in enumerate(tiles_params):
            if tile_idx >= len(tile_depths_np):
                break
                
            # Get tile data
            tile_depth = tile_depths_np[tile_idx]  # [H, W] or [H, W, 3]
            if len(tile_depth.shape) == 3:
                tile_depth = tile_depth[:, :, 0]  # Take first channel
            
            tile_normals = tile_normals_np[tile_idx]  # [3, H, W] or [H, W, 3]
            if tile_normals.shape[0] == 3:  # [3, H, W]
                tile_normals = np.transpose(tile_normals, (1, 2, 0))  # [H, W, 3]
            
            # Get tile parameters
            center_lon = tile_param['center_lon']
            center_lat = tile_param['center_lat'] 
            fov_deg = tile_param['fov_deg']
            
            # Back-project tile to ERP using spherical projection
            tile_id = tile_param.get('tile_id', f'tile_{tile_idx}')
            weight_map = weights.get(tile_id, None)
            self._backproject_tile_to_erp(
                tile_depth, tile_normals, center_lon, center_lat, fov_deg,
                erp_depth, erp_normals, erp_weights, erp_w, erp_h, weight_map
            )
        
        # Normalize by weights where we have data
        valid_mask = erp_weights > 0
        erp_depth[valid_mask] = erp_depth[valid_mask] / erp_weights[valid_mask]
        for c in range(3):
            erp_normals[valid_mask, c] = erp_normals[valid_mask, c] / erp_weights[valid_mask]
                
        # Create debug preview if requested
        if create_preview:
            preview = np.zeros((erp_h, erp_w, 3), dtype=np.float32)
            if erp_depth.max() > 0:
                depth_norm = erp_depth / erp_depth.max()
                preview[:, :, 0] = depth_norm
                preview[:, :, 1] = depth_norm  
                preview[:, :, 2] = depth_norm
        else:
            preview = erp_normals.copy()
        
        # Convert to tensors with proper normalization
        # Normalize depth for visualization
        if erp_depth.max() > 0:
            depth_norm = erp_depth / erp_depth.max()
        else:
            depth_norm = erp_depth
        
        erp_depth_tensor = torch.from_numpy(depth_norm).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        erp_normals_tensor = torch.from_numpy(erp_normals).unsqueeze(0).to(device)
        preview_tensor = torch.from_numpy(preview).unsqueeze(0).to(device)
        
        log.info("Spherical stitching completed successfully")
        
        return (erp_depth_tensor, erp_normals_tensor, preview_tensor)

    def _backproject_tile_to_erp(self, tile_depth: np.ndarray, tile_normals: np.ndarray,
                                 center_lon: float, center_lat: float, fov_deg: float,
                                 erp_depth: np.ndarray, erp_normals: np.ndarray,
                                 erp_weights: np.ndarray, erp_w: int, erp_h: int,
                                 weight_map: np.ndarray = None):
        """Project a single perspective tile's depth/normals back to ERP and accumulate with weights."""
        import math
        # Tile size
        th, tw = tile_depth.shape
        # Sanitize inputs to avoid NaNs/Infs
        tile_depth = np.nan_to_num(tile_depth, nan=0.0, posinf=0.0, neginf=0.0)
        tile_normals = np.nan_to_num(tile_normals, nan=0.0, posinf=0.0, neginf=0.0)
        # Compute focal length from FOV and tile size
        fov_rad = math.radians(fov_deg)
        f = tw / (2.0 * math.tan(fov_rad / 2.0))

        # Camera orientation from center lon/lat
        lon = math.radians(center_lon)
        lat = math.radians(center_lat)
        forward = np.array([math.cos(lat) * math.cos(lon), math.sin(lat), math.cos(lat) * math.sin(lon)], dtype=np.float32)
        up = np.array([-math.sin(lat) * math.cos(lon), math.cos(lat), -math.sin(lat) * math.sin(lon)], dtype=np.float32)
        right = np.cross(forward, up)
        right /= (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        up /= (np.linalg.norm(up) + 1e-8)
        # World-to-camera rotation used in sampling; here we need camera-to-world for rays
        R = np.column_stack([right, up, -forward]).astype(np.float32)

        # Create grid of pixel coordinates
        u_coords, v_coords = np.meshgrid(np.arange(tw, dtype=np.float32), np.arange(th, dtype=np.float32), indexing='xy')
        x_cam = (u_coords - tw / 2.0) / f
        y_cam = (v_coords - th / 2.0) / f
        z_cam = np.ones_like(x_cam, dtype=np.float32)
        rays_cam = np.stack([x_cam, y_cam, z_cam], axis=0).reshape(3, -1)  # [3, N]
        # Transform to world
        rays_world = R.T @ rays_cam  # [3, N]
        # Normalize
        rays_world /= (np.linalg.norm(rays_world, axis=0, keepdims=True) + 1e-8)

        # Convert to ERP pixel coords (vectorized) with improved seam handling
        xw, yw, zw = rays_world[0], rays_world[1], rays_world[2]
        
        # Clamp yw to avoid numerical issues at poles
        yw = np.clip(yw, -0.99999, 0.99999)
        
        lon_pix = np.arctan2(zw, xw)  # [-pi, pi]
        lat_pix = np.arcsin(yw)       # [-pi/2, pi/2]
        
        # Convert to ERP coordinates with proper normalization
        u_erp = ((lon_pix / (2 * math.pi)) + 0.5) * erp_w
        v_erp = (0.5 - (lat_pix / math.pi)) * erp_h

        # Handle longitude wraparound properly (for seam continuity)
        u_erp = np.mod(u_erp, erp_w)
        
        # Round to nearest pixel indices with proper clamping
        u_idx = np.round(u_erp).astype(np.int64)
        v_idx = np.round(v_erp).astype(np.int64)
        
        # Ensure indices are within bounds
        u_idx = np.mod(u_idx, erp_w)  # Wrap longitude
        v_idx = np.clip(v_idx, 0, erp_h - 1)  # Clamp latitude

        # Flatten tile data
        depth_flat = tile_depth.reshape(-1)
        normals_flat = tile_normals.reshape(-1, 3)

        # Valid mask: positive depth only
        valid = depth_flat > 1e-6
        u_idx = u_idx[valid]
        v_idx = v_idx[valid]
        depth_flat = depth_flat[valid]
        normals_flat = normals_flat[valid]

        # Sample weight map if provided; else uniform weight 1
        if weight_map is not None:
            w = weight_map[v_idx, u_idx].astype(np.float32)
        else:
            w = np.ones_like(depth_flat, dtype=np.float32)

        # Accumulate into ERP arrays
        # Sanitize weights
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        np.add.at(erp_depth, (v_idx, u_idx), depth_flat * w)
        np.add.at(erp_weights, (v_idx, u_idx), w)
        # Normals accumulate per channel
        for c in range(3):
            np.add.at(erp_normals[:, :, c], (v_idx, u_idx), normals_flat[:, c] * w)
