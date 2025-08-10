"""
Spherical projection utilities for equirectangular panorama processing.
Handles conversion between ERP coordinates and 3D spherical coordinates.
"""

import numpy as np
import torch
import cv2
from typing import Tuple, List, Dict, Any
import math

class SphericalProjection:
    """Handles spherical projection operations for equirectangular panoramas."""
    
    @staticmethod
    def erp_to_xyz(u: float, v: float, width: int, height: int) -> Tuple[float, float, float]:
        """Convert ERP pixel coordinates to 3D unit sphere coordinates.
        
        Args:
            u, v: Pixel coordinates in ERP image
            width, height: ERP image dimensions
            
        Returns:
            x, y, z: 3D coordinates on unit sphere
        """
        # Normalize to [0, 1]
        lon_norm = u / width   # 0 to 1
        lat_norm = v / height  # 0 to 1
        
        # Convert to spherical coordinates
        longitude = (lon_norm - 0.5) * 2 * math.pi  # -π to π
        latitude = (0.5 - lat_norm) * math.pi       # -π/2 to π/2
        
        # Convert to Cartesian coordinates
        x = math.cos(latitude) * math.cos(longitude)
        y = math.sin(latitude)
        z = math.cos(latitude) * math.sin(longitude)
        
        return x, y, z
    
    @staticmethod
    def xyz_to_erp(x: float, y: float, z: float, width: int, height: int) -> Tuple[float, float]:
        """Convert 3D unit sphere coordinates to ERP pixel coordinates.
        
        Args:
            x, y, z: 3D coordinates on unit sphere
            width, height: ERP image dimensions
            
        Returns:
            u, v: Pixel coordinates in ERP image
        """
        # Convert to spherical coordinates
        longitude = math.atan2(z, x)  # -π to π
        latitude = math.asin(y)       # -π/2 to π/2
        
        # Normalize and convert to pixel coordinates
        lon_norm = (longitude / (2 * math.pi)) + 0.5  # 0 to 1
        lat_norm = 0.5 - (latitude / math.pi)         # 0 to 1
        
        u = lon_norm * width
        v = lat_norm * height
        
        return u, v

    @staticmethod
    def sample_perspective_tile(erp_image: np.ndarray, center_lon: float, center_lat: float, 
                              fov_deg: float, tile_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a perspective tile from equirectangular panorama.
        
        Args:
            erp_image: Input ERP image [H, W, C]
            center_lon: Center longitude in degrees [-180, 180]
            center_lat: Center latitude in degrees [-90, 90] 
            fov_deg: Field of view in degrees
            tile_size: Output tile resolution
            
        Returns:
            tile_image: Perspective tile [tile_size, tile_size, C]
            intrinsics: Camera intrinsics matrix [3, 3]
            extrinsics: Camera extrinsics matrix [4, 4]
        """
        height, width = erp_image.shape[:2]
        
        # Convert angles to radians
        center_lon_rad = math.radians(center_lon)
        center_lat_rad = math.radians(center_lat)
        fov_rad = math.radians(fov_deg)
        
        # Calculate focal length for perspective projection
        focal_length = tile_size / (2 * math.tan(fov_rad / 2))
        
        # Create perspective camera intrinsics
        intrinsics = np.array([
            [focal_length, 0, tile_size / 2],
            [0, focal_length, tile_size / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Create camera rotation matrix (look-at transformation)
        # Camera points towards (center_lon, center_lat)
        forward = np.array([
            math.cos(center_lat_rad) * math.cos(center_lon_rad),
            math.sin(center_lat_rad),
            math.cos(center_lat_rad) * math.sin(center_lon_rad)
        ])
        
        # Up vector (towards north pole, adjusted for camera orientation)
        up = np.array([
            -math.sin(center_lat_rad) * math.cos(center_lon_rad),
            math.cos(center_lat_rad),
            -math.sin(center_lat_rad) * math.sin(center_lon_rad)
        ])
        
        # Right vector (east direction)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create rotation matrix (world to camera)
        R = np.column_stack([right, up, -forward])
        
        # Create extrinsics matrix (camera at origin)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R
        
        # Sample perspective tile using inverse projection (vectorized for speed)
        tile_image = np.zeros((tile_size, tile_size, erp_image.shape[2]), dtype=erp_image.dtype)
        
        # Create coordinate grids for vectorized processing
        u_coords, v_coords = np.meshgrid(np.arange(tile_size), np.arange(tile_size), indexing='xy')
        
        # Convert tile pixels to normalized camera coordinates (vectorized)
        x_cam = (u_coords - tile_size / 2) / focal_length
        y_cam = (v_coords - tile_size / 2) / focal_length
        z_cam = np.ones_like(x_cam)
        
        # Stack camera rays [3, H, W]
        cam_rays = np.stack([x_cam.flatten(), y_cam.flatten(), z_cam.flatten()], axis=0)  # [3, N]
        
        # Transform to world coordinates (batch operation)
        world_rays = R.T @ cam_rays  # [3, N]
        
        # Normalize rays
        ray_norms = np.linalg.norm(world_rays, axis=0, keepdims=True)
        world_rays = world_rays / (ray_norms + 1e-8)
        
        # Convert to ERP coordinates (vectorized)
        x_world, y_world, z_world = world_rays[0], world_rays[1], world_rays[2]
        
        # Batch conversion to ERP coordinates
        erp_coords = np.zeros((2, len(x_world)))
        for i in range(len(x_world)):
            erp_u, erp_v = SphericalProjection.xyz_to_erp(
                x_world[i], y_world[i], z_world[i], width, height
            )
            erp_coords[0, i] = erp_u
            erp_coords[1, i] = erp_v
        
        # Handle longitude wrapping and clamping
        erp_coords[0] = np.where(erp_coords[0] < 0, erp_coords[0] + width, erp_coords[0])
        erp_coords[0] = np.where(erp_coords[0] >= width, erp_coords[0] - width, erp_coords[0])
        erp_coords[1] = np.clip(erp_coords[1], 0, height - 1)
        
        # Bilinear sampling (vectorized where possible)
        for idx, (v, u) in enumerate(zip(v_coords.flatten(), u_coords.flatten())):
            erp_u, erp_v = erp_coords[0, idx], erp_coords[1, idx]
            
            if 0 <= erp_u < width and 0 <= erp_v < height:
                u0, v0 = int(erp_u), int(erp_v)
                u1, v1 = min(u0 + 1, width - 1), min(v0 + 1, height - 1)
                
                # Handle longitude wrap for u1
                if u1 == width:
                    u1 = 0
                
                wu = erp_u - u0
                wv = erp_v - v0
                
                tile_image[v, u] = (
                    (1 - wu) * (1 - wv) * erp_image[v0, u0] +
                    wu * (1 - wv) * erp_image[v0, u1] +
                    (1 - wu) * wv * erp_image[v1, u0] +
                    wu * wv * erp_image[v1, u1]
                )
        
        return tile_image, intrinsics, extrinsics

    @staticmethod
    def create_tile_grid(grid_yaw: int = 6, grid_pitch: int = 3, 
                        fov_deg: float = 100, overlap_deg: float = 15) -> List[Dict[str, float]]:
        """Create a grid of perspective tile parameters for spherical sampling.
        
        Args:
            grid_yaw: Number of tiles in yaw (longitude) direction
            grid_pitch: Number of tiles in pitch (latitude) direction  
            fov_deg: Field of view for each tile
            overlap_deg: Overlap between adjacent tiles
            
        Returns:
            List of tile parameters with center_lon, center_lat, fov_deg
        """
        tiles = []
        
        # Calculate tile spacing
        yaw_step = 360.0 / grid_yaw
        pitch_range = 180.0 - 2 * overlap_deg  # Avoid poles
        pitch_step = pitch_range / max(1, grid_pitch - 1)
        
        for i in range(grid_yaw):
            center_lon = (i * yaw_step) - 180.0  # -180 to 180
            
            for j in range(grid_pitch):
                if grid_pitch == 1:
                    center_lat = 0.0
                else:
                    center_lat = 90.0 - overlap_deg - (j * pitch_step)
                
                tiles.append({
                    'center_lon': center_lon,
                    'center_lat': center_lat,
                    'fov_deg': fov_deg,
                    'tile_id': f'tile_{i}_{j}'
                })
        
        return tiles

    @staticmethod
    def calculate_stitch_weights(tiles_params: List[Dict], width: int, height: int) -> Dict[str, np.ndarray]:
        """Calculate blending weights for stitching tiles back to ERP.
        
        Args:
            tiles_params: List of tile parameters
            width, height: ERP dimensions
            
        Returns:
            Dictionary mapping tile_id to weight maps
        """
        # Initialize accumulation arrays
        total_weight = np.zeros((height, width), dtype=np.float32)
        tile_weights = {}
        
        for tile_params in tiles_params:
            tile_id = tile_params['tile_id']
            center_lon = tile_params['center_lon'] 
            center_lat = tile_params['center_lat']
            fov_deg = tile_params['fov_deg']
            
            # Create weight map for this tile
            weight_map = np.zeros((height, width), dtype=np.float32)
            
            # Calculate angular distance from tile center for each ERP pixel
            for v in range(height):
                for u in range(width):
                    # Convert ERP pixel to spherical coordinates
                    x, y, z = SphericalProjection.erp_to_xyz(u, v, width, height)
                    pixel_lon = math.degrees(math.atan2(z, x))
                    pixel_lat = math.degrees(math.asin(y))
                    
                    # Calculate angular distance from tile center
                    # Use great circle distance
                    center_lon_rad = math.radians(center_lon)
                    center_lat_rad = math.radians(center_lat)
                    pixel_lon_rad = math.radians(pixel_lon)
                    pixel_lat_rad = math.radians(pixel_lat)
                    
                    cos_dist = (math.sin(center_lat_rad) * math.sin(pixel_lat_rad) +
                              math.cos(center_lat_rad) * math.cos(pixel_lat_rad) *
                              math.cos(abs(pixel_lon_rad - center_lon_rad)))
                    
                    # Clamp to avoid numerical issues
                    cos_dist = max(-1.0, min(1.0, cos_dist))
                    angular_dist_deg = math.degrees(math.acos(cos_dist))
                    
                    # Create smooth falloff within FOV
                    if angular_dist_deg < fov_deg / 2:
                        # Smooth cosine falloff
                        normalized_dist = angular_dist_deg / (fov_deg / 2)
                        weight = 0.5 * (1 + math.cos(math.pi * normalized_dist))
                        weight_map[v, u] = weight
            
            tile_weights[tile_id] = weight_map
            total_weight += weight_map
        
        # Normalize weights to sum to 1
        for tile_id in tile_weights:
            tile_weights[tile_id] = np.divide(
                tile_weights[tile_id], 
                total_weight, 
                out=np.zeros_like(tile_weights[tile_id]), 
                where=total_weight != 0
            )
        
        return tile_weights