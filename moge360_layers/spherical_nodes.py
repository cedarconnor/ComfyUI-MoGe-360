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
                "erp_image": ("IMAGE",),
                "grid_yaw": ("INT", {"default": 6, "min": 2, "max": 12, "step": 1}),
                "grid_pitch": ("INT", {"default": 3, "min": 1, "max": 6, "step": 1}),
                "tile_fov": ("FLOAT", {"default": 100.0, "min": 60.0, "max": 140.0, "step": 5.0}),
                "overlap": ("FLOAT", {"default": 15.0, "min": 5.0, "max": 30.0, "step": 1.0}),
                "tile_resolution": ("INT", {"default": 768, "min": 512, "max": 1024, "step": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STITCH_MAP", "TILE_PARAMS")
    RETURN_NAMES = ("perspective_tiles", "stitch_map", "tile_params")
    FUNCTION = "sample_tiles"
    CATEGORY = "MoGe360/Core"
    DESCRIPTION = "Sample perspective tiles from ERP panorama with spherical projection"

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
            'overlap_deg': overlap
        }
        
        return (tiles_tensor, stitch_map, tile_params_out)


class MoGe_PerTile_Geometry:
    """Process perspective tiles through MoGe model to get depth and normals."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_model": ("MOGEMODEL",),
                "perspective_tiles": ("IMAGE",),
                "tile_params": ("TILE_PARAMS",),
                "resolution_level": ("INT", {"default": 9, "min": 6, "max": 11, "step": 1}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "lock_fov": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "TILE_GEOMETRY")
    RETURN_NAMES = ("tile_depths", "tile_normals", "tile_geometry")
    FUNCTION = "process_tiles"
    CATEGORY = "MoGe360/Core"
    DESCRIPTION = "Process perspective tiles through MoGe to get depth and normals"

    def process_tiles(self, moge_model, perspective_tiles: torch.Tensor, tile_params: Dict,
                     resolution_level: int, batch_size: int, lock_fov: bool):
        
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        
        N, H, W, C = perspective_tiles.shape
        
        # Process tiles in batches
        all_depths = []
        all_normals = []
        all_points = []
        all_masks = []
        
        log.info(f"Processing {N} tiles in batches of {batch_size}")
        
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            batch_tiles = perspective_tiles[i:batch_end]
            
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
                
                # Calculate normals from points if available
                if 'normals' in output:
                    normals = output['normals']  # [3, H, W]
                    # Also normalize normals shape if needed
                    if normals.dim() == 3 and normals.shape[2] == 3 and normals.shape[0] != 3:
                        normals = normals.permute(2, 0, 1)
                        log.info(f"Converted normals from [H, W, 3] to [3, H, W]: {normals.shape}")
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
        """Compute surface normals from 3D points using gradient method."""
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
        
        # Compute gradients for each coordinate
        # X gradients
        dx_du = torch.zeros_like(x)
        dx_dv = torch.zeros_like(x)
        dx_du[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2
        dx_dv[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2
        # Handle edges
        dx_du[:, 0] = x[:, 1] - x[:, 0]
        dx_du[:, -1] = x[:, -1] - x[:, -2]
        dx_dv[0, :] = x[1, :] - x[0, :]
        dx_dv[-1, :] = x[-1, :] - x[-2, :]
        
        # Y gradients  
        dy_du = torch.zeros_like(y)
        dy_dv = torch.zeros_like(y)
        dy_du[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2
        dy_dv[1:-1, :] = (y[2:, :] - y[:-2, :]) / 2
        # Handle edges
        dy_du[:, 0] = y[:, 1] - y[:, 0]
        dy_du[:, -1] = y[:, -1] - y[:, -2]
        dy_dv[0, :] = y[1, :] - y[0, :]
        dy_dv[-1, :] = y[-1, :] - y[-2, :]
        
        # Z gradients
        dz_du = torch.zeros_like(z)
        dz_dv = torch.zeros_like(z)
        dz_du[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / 2
        dz_dv[1:-1, :] = (z[2:, :] - z[:-2, :]) / 2
        # Handle edges
        dz_du[:, 0] = z[:, 1] - z[:, 0]
        dz_du[:, -1] = z[:, -1] - z[:, -2]
        dz_dv[0, :] = z[1, :] - z[0, :]
        dz_dv[-1, :] = z[-1, :] - z[-2, :]
        
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
        
        # Normalize
        norm = torch.sqrt(normals[0]**2 + normals[1]**2 + normals[2]**2).unsqueeze(0)  # [1, H, W]
        norm = torch.clamp(norm, min=1e-8)  # Avoid division by zero
        normals = normals / norm
        
        log.info(f"Computed normals with shape: {normals.shape}")
        return normals


class Depth_Normal_Stitcher_360:
    """Stitch perspective tile depth and normals back to equirectangular format."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_geometry": ("TILE_GEOMETRY",),
                "stitch_map": ("STITCH_MAP",),
                "blend_mode": (["weighted", "feather", "poisson"], {"default": "weighted"}),
                "edge_feather": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("erp_depth", "erp_normals", "erp_mask")
    FUNCTION = "stitch_geometry"
    CATEGORY = "MoGe360/Core" 
    DESCRIPTION = "Stitch tile depth and normals back to equirectangular panorama"

    def stitch_geometry(self, tile_geometry: Dict, stitch_map: Dict, 
                       blend_mode: str, edge_feather: float):
        
        device = mm.get_torch_device()
        
        # Extract data
        tile_depths = tile_geometry['depths']  # [N, H, W]
        tile_normals = tile_geometry['normals']  # [N, 3, H, W]
        tile_masks = tile_geometry['masks']  # [N, H, W]
        tiles_params = tile_geometry['tile_params']['tiles_params']
        
        erp_h, erp_w = stitch_map['erp_dimensions']
        tile_res = stitch_map['tile_resolution']
        weights = stitch_map['weights']
        
        # Initialize output arrays
        erp_depth = np.zeros((erp_h, erp_w), dtype=np.float32)
        erp_normals = np.zeros((erp_h, erp_w, 3), dtype=np.float32)
        erp_mask = np.zeros((erp_h, erp_w), dtype=np.float32)
        
        log.info(f"Stitching {len(tiles_params)} tiles to {erp_w}x{erp_h} ERP")
        
        # Process each tile
        for i, tile_params in enumerate(tiles_params):
            tile_id = tile_params['tile_id']
            
            if tile_id not in weights:
                continue
                
            log.info(f"Stitching tile {i+1}/{len(tiles_params)}: {tile_id}")
            
            weight_map = weights[tile_id]
            tile_depth = tile_depths[i].numpy()  # [H, W]
            
            # Clean depth data - remove NaNs and infinities
            tile_depth = np.nan_to_num(tile_depth, nan=0.0, posinf=10.0, neginf=0.0)
            
            # Handle normals shape conversion safely
            tile_normal_tensor = tile_normals[i]  # Should be [3, H, W]
            if tile_normal_tensor.shape[0] == 3:
                tile_normal = tile_normal_tensor.permute(1, 2, 0).numpy()  # [H, W, 3]
            else:
                log.error(f"Unexpected normals shape for tile {i}: {tile_normal_tensor.shape}")
                tile_normal = np.zeros((tile_depth.shape[0], tile_depth.shape[1], 3), dtype=np.float32)
            
            # Clean normals data
            tile_normal = np.nan_to_num(tile_normal, nan=0.0, posinf=1.0, neginf=-1.0)
            
            tile_mask = tile_masks[i].numpy()  # [H, W]
            tile_mask = np.nan_to_num(tile_mask, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Reverse perspective projection to map tile back to ERP
            self._stitch_tile_to_erp(
                tile_depth, tile_normal, tile_mask, weight_map,
                tile_params, tile_res, erp_depth, erp_normals, erp_mask
            )
        
        # Convert to tensors
        erp_depth_tensor = torch.from_numpy(erp_depth).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        erp_normals_tensor = torch.from_numpy(erp_normals).unsqueeze(0).to(device)
        erp_mask_tensor = torch.from_numpy(erp_mask).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        
        log.info("Stitching completed successfully")
        
        return (erp_depth_tensor, erp_normals_tensor, erp_mask_tensor)
    
    def _stitch_tile_to_erp(self, tile_depth: np.ndarray, tile_normal: np.ndarray, 
                           tile_mask: np.ndarray, weight_map: np.ndarray,
                           tile_params: Dict, tile_res: int,
                           erp_depth: np.ndarray, erp_normals: np.ndarray, erp_mask: np.ndarray):
        """Stitch a single tile back to ERP using reverse perspective projection."""
        
        center_lon = tile_params['center_lon']
        center_lat = tile_params['center_lat'] 
        fov_deg = tile_params['fov_deg']
        
        erp_h, erp_w = weight_map.shape
        
        # Create perspective camera parameters
        focal_length = tile_res / (2 * np.tan(np.radians(fov_deg / 2)))
        
        # Count pixels with significant weight to estimate progress
        significant_pixels = np.sum(weight_map > 1e-6)
        processed_pixels = 0
        progress_interval = max(1, significant_pixels // 10)  # Update every 10%
        
        # For each ERP pixel that has weight for this tile
        for v in range(erp_h):
            for u in range(erp_w):
                if weight_map[v, u] <= 1e-6:
                    continue
                
                processed_pixels += 1
                if processed_pixels % progress_interval == 0:
                    progress_pct = (processed_pixels / significant_pixels) * 100
                    log.info(f"  Stitching progress: {progress_pct:.1f}% ({processed_pixels}/{significant_pixels} pixels)")
                    
                # Convert ERP pixel to 3D direction
                x, y, z = SphericalProjection.erp_to_xyz(u, v, erp_w, erp_h)
                world_point = np.array([x, y, z])
                
                # Project to tile camera coordinates
                tile_u, tile_v = self._project_to_tile(
                    world_point, center_lon, center_lat, focal_length, tile_res
                )
                
                # Check if projection is within tile bounds with margin
                if 1 <= tile_u < tile_res - 1 and 1 <= tile_v < tile_res - 1:
                    # Bilinear interpolation from tile
                    u0, v0 = int(tile_u), int(tile_v)
                    u1, v1 = u0 + 1, v0 + 1
                    
                    # Additional bounds check
                    if u1 >= tile_res or v1 >= tile_res:
                        continue
                    
                    wu = tile_u - u0
                    wv = tile_v - v0
                    
                    # Bounds check before accessing arrays
                    if v1 < tile_depth.shape[0] and u1 < tile_depth.shape[1]:
                        # Interpolate depth
                        interp_depth = (
                            (1 - wu) * (1 - wv) * tile_depth[v0, u0] +
                            wu * (1 - wv) * tile_depth[v0, u1] +
                            (1 - wu) * wv * tile_depth[v1, u0] +
                            wu * wv * tile_depth[v1, u1]
                        )
                        
                        # Interpolate normals (tile_normal is [H, W, 3])
                        if v1 < tile_normal.shape[0] and u1 < tile_normal.shape[1]:
                            interp_normal = (
                                (1 - wu) * (1 - wv) * tile_normal[v0, u0, :] +
                                wu * (1 - wv) * tile_normal[v0, u1, :] +
                                (1 - wu) * wv * tile_normal[v1, u0, :] +
                                wu * wv * tile_normal[v1, u1, :]
                            )
                        else:
                            interp_normal = np.zeros(3, dtype=np.float32)
                        
                        # Interpolate mask
                        interp_mask = (
                            (1 - wu) * (1 - wv) * tile_mask[v0, u0] +
                            wu * (1 - wv) * tile_mask[v0, u1] +
                            (1 - wu) * wv * tile_mask[v1, u0] +
                            wu * wv * tile_mask[v1, u1]
                        )
                        
                        # Weighted accumulation with NaN checking
                        weight = weight_map[v, u] * interp_mask
                        if not np.isnan(weight) and not np.isinf(weight):
                            if not np.isnan(interp_depth) and not np.isinf(interp_depth):
                                erp_depth[v, u] += weight * interp_depth
                            if not np.any(np.isnan(interp_normal)) and not np.any(np.isinf(interp_normal)):
                                erp_normals[v, u] += weight * interp_normal
                            erp_mask[v, u] += weight
    
    def _project_to_tile(self, world_point: np.ndarray, center_lon: float, center_lat: float,
                        focal_length: float, tile_res: int) -> Tuple[float, float]:
        """Project world point to tile camera coordinates."""
        
        # Create camera rotation matrix
        center_lon_rad = np.radians(center_lon)
        center_lat_rad = np.radians(center_lat)
        
        # Camera forward direction
        forward = np.array([
            np.cos(center_lat_rad) * np.cos(center_lon_rad),
            np.sin(center_lat_rad),
            np.cos(center_lat_rad) * np.sin(center_lon_rad)
        ])
        
        # Up and right vectors
        up = np.array([
            -np.sin(center_lat_rad) * np.cos(center_lon_rad),
            np.cos(center_lat_rad),
            -np.sin(center_lat_rad) * np.sin(center_lon_rad)
        ])
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # World to camera transformation
        R = np.column_stack([right, up, -forward])
        cam_point = R @ world_point
        
        # Perspective projection
        if cam_point[2] <= 0:  # Behind camera
            return -1, -1
        
        x_proj = focal_length * cam_point[0] / cam_point[2]
        y_proj = focal_length * cam_point[1] / cam_point[2]
        
        # Convert to pixel coordinates
        tile_u = x_proj + tile_res / 2
        tile_v = y_proj + tile_res / 2
        
        return tile_u, tile_v