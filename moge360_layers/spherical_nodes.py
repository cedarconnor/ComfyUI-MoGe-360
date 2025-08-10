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
        
        # Sample all tiles
        tiles = []
        intrinsics_list = []
        extrinsics_list = []
        
        for tile_params in tiles_params:
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
                points = output['points']  # [3, H, W] 
                mask = output['mask']  # [H, W]
                
                # Calculate normals from points if available
                if 'normals' in output:
                    normals = output['normals']  # [3, H, W]
                else:
                    # Compute normals from depth/points
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
        # points: [3, H, W]
        C, H, W = points.shape
        
        # Compute gradients
        grad_x = torch.zeros_like(points)
        grad_y = torch.zeros_like(points)
        
        # Central differences
        grad_x[:, :, 1:-1] = (points[:, :, 2:] - points[:, :, :-2]) / 2
        grad_y[:, 1:-1, :] = (points[:, 2:, :] - points[:, :-2, :]) / 2
        
        # Handle borders with forward/backward differences
        grad_x[:, :, 0] = points[:, :, 1] - points[:, :, 0]
        grad_x[:, :, -1] = points[:, :, -1] - points[:, :, -2]
        grad_y[:, 0, :] = points[:, 1, :] - points[:, 0, :]
        grad_y[:, -1, :] = points[:, -1, :] - points[:, -2, :]
        
        # Cross product to get normals
        normals = torch.cross(grad_x, grad_y, dim=0)
        
        # Normalize
        norm = torch.norm(normals, dim=0, keepdim=True)
        normals = normals / (norm + 1e-8)
        
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
                
            weight_map = weights[tile_id]
            tile_depth = tile_depths[i].numpy()  # [H, W]
            tile_normal = tile_normals[i].permute(1, 2, 0).numpy()  # [H, W, 3]
            tile_mask = tile_masks[i].numpy()  # [H, W]
            
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
        
        # For each ERP pixel that has weight for this tile
        for v in range(erp_h):
            for u in range(erp_w):
                if weight_map[v, u] <= 1e-6:
                    continue
                    
                # Convert ERP pixel to 3D direction
                x, y, z = SphericalProjection.erp_to_xyz(u, v, erp_w, erp_h)
                world_point = np.array([x, y, z])
                
                # Project to tile camera coordinates
                tile_u, tile_v = self._project_to_tile(
                    world_point, center_lon, center_lat, focal_length, tile_res
                )
                
                # Check if projection is within tile bounds
                if 0 <= tile_u < tile_res and 0 <= tile_v < tile_res:
                    # Bilinear interpolation from tile
                    u0, v0 = int(tile_u), int(tile_v)
                    u1, v1 = min(u0 + 1, tile_res - 1), min(v0 + 1, tile_res - 1)
                    
                    wu = tile_u - u0
                    wv = tile_v - v0
                    
                    # Interpolate depth
                    interp_depth = (
                        (1 - wu) * (1 - wv) * tile_depth[v0, u0] +
                        wu * (1 - wv) * tile_depth[v0, u1] +
                        (1 - wu) * wv * tile_depth[v1, u0] +
                        wu * wv * tile_depth[v1, u1]
                    )
                    
                    # Interpolate normals
                    interp_normal = (
                        (1 - wu) * (1 - wv) * tile_normal[v0, u0] +
                        wu * (1 - wv) * tile_normal[v0, u1] +
                        (1 - wu) * wv * tile_normal[v1, u0] +
                        wu * wv * tile_normal[v1, u1]
                    )
                    
                    # Interpolate mask
                    interp_mask = (
                        (1 - wu) * (1 - wv) * tile_mask[v0, u0] +
                        wu * (1 - wv) * tile_mask[v0, u1] +
                        (1 - wu) * wv * tile_mask[v1, u0] +
                        wu * wv * tile_mask[v1, u1]
                    )
                    
                    # Weighted accumulation
                    weight = weight_map[v, u] * interp_mask
                    
                    erp_depth[v, u] += weight * interp_depth
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