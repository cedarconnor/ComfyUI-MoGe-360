"""
Layer processing nodes for 360° panorama decomposition.
Implements sky/background splitting and layer building with ERP-aware processing.
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Any
import logging

from .spherical_utils import SphericalProjection
import comfy.model_management as mm

log = logging.getLogger(__name__)

class Sky_Background_Splitter:
    """Split equirectangular panorama into sky and background layers."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "erp_image": ("IMAGE",),  # [1, H, W, 3]
                "method": (["gradient", "blue_sky", "horizon_line", "auto"], {"default": "auto"}),
                "sky_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "horizon_latitude": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0, "step": 1.0}),
                "feather_amount": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.2, "step": 0.01}),
                "min_sky_height": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.8, "step": 0.01}),
            },
            "optional": {
                "sky_mask_hint": ("IMAGE",),  # Optional manual sky mask
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("sky_mask", "background_mask", "combined_preview")
    FUNCTION = "split_sky_background"
    CATEGORY = "MoGe360/Layers"
    DESCRIPTION = "Split ERP panorama into sky and background/foreground layers"

    def split_sky_background(self, erp_image: torch.Tensor, method: str, sky_threshold: float,
                           horizon_latitude: float, feather_amount: float, min_sky_height: float,
                           sky_mask_hint: torch.Tensor = None):
        
        device = mm.get_torch_device()
        B, H, W, C = erp_image.shape
        
        # Convert to numpy for processing
        img_np = erp_image[0].cpu().numpy()  # [H, W, 3]
        
        log.info(f"Splitting sky/background for {W}x{H} ERP using method: {method}")
        
        # Create base sky mask using selected method
        if method == "auto":
            sky_mask = self._auto_sky_detection(img_np, sky_threshold, horizon_latitude, min_sky_height)
        elif method == "gradient":
            sky_mask = self._gradient_sky_detection(img_np, sky_threshold, horizon_latitude)
        elif method == "blue_sky":
            sky_mask = self._blue_sky_detection(img_np, sky_threshold)
        elif method == "horizon_line":
            sky_mask = self._horizon_line_detection(img_np, horizon_latitude, min_sky_height)
        else:
            raise ValueError(f"Unknown sky detection method: {method}")
        
        # Refine with manual hint if provided
        if sky_mask_hint is not None:
            hint_mask = sky_mask_hint[0, :, :, 0].cpu().numpy()  # [H, W]
            sky_mask = self._combine_with_hint(sky_mask, hint_mask, sky_threshold)
        
        # Apply feathering for smooth transitions
        if feather_amount > 0:
            sky_mask = self._apply_feathering(sky_mask, feather_amount, H, W)
        
        # Ensure ERP seam continuity
        sky_mask = self._fix_erp_seam(sky_mask)
        
        # Create background mask (inverse of sky)
        background_mask = 1.0 - sky_mask
        
        # Create visualization
        combined_preview = self._create_preview(img_np, sky_mask, background_mask)
        
        # Convert back to tensors
        sky_tensor = torch.from_numpy(sky_mask).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        bg_tensor = torch.from_numpy(background_mask).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        preview_tensor = torch.from_numpy(combined_preview).unsqueeze(0).to(device)
        
        log.info(f"Sky coverage: {np.mean(sky_mask):.2%}, Background coverage: {np.mean(background_mask):.2%}")
        
        return (sky_tensor, bg_tensor, preview_tensor)
    
    def _auto_sky_detection(self, img: np.ndarray, threshold: float, horizon_lat: float, min_sky_height: float) -> np.ndarray:
        """Automatic sky detection combining multiple methods."""
        H, W = img.shape[:2]
        
        log.info(f"Auto sky detection - threshold: {threshold}, horizon: {horizon_lat}, min_height: {min_sky_height}")
        
        # Combine multiple detection methods
        gradient_mask = self._gradient_sky_detection(img, threshold, horizon_lat)
        blue_mask = self._blue_sky_detection(img, threshold * 0.8)  # More lenient for blue
        horizon_mask = self._horizon_line_detection(img, horizon_lat, min_sky_height)
        
        log.info(f"Detection results - gradient: {gradient_mask.sum():.0f}, blue: {blue_mask.sum():.0f}, horizon: {horizon_mask.sum():.0f}")
        
        # Weighted combination
        combined_mask = (
            0.4 * gradient_mask +
            0.3 * blue_mask +
            0.3 * horizon_mask
        )
        
        # More aggressive thresholding for sky detection
        adjusted_threshold = min(threshold, 0.3)  # Lower threshold for better detection
        sky_mask = (combined_mask > adjusted_threshold).astype(np.float32)
        
        log.info(f"Combined mask sum: {combined_mask.sum():.0f}, final sky pixels: {sky_mask.sum():.0f}")
        
        return sky_mask
    
    def _gradient_sky_detection(self, img: np.ndarray, threshold: float, horizon_lat: float) -> np.ndarray:
        """Detect sky using vertical brightness gradient."""
        H, W = img.shape[:2]
        
        # Convert to grayscale
        gray = np.mean(img, axis=2)
        
        # Calculate vertical gradient (sky is typically brighter at top)
        gradient_y = np.gradient(gray, axis=0)
        
        # Normalize and threshold
        grad_norm = (gradient_y - gradient_y.min()) / (gradient_y.max() - gradient_y.min() + 1e-8)
        
        # Sky is where gradient is negative (brightness decreases downward)
        sky_indicator = np.clip(-gradient_y, 0, None)
        sky_indicator = sky_indicator / (sky_indicator.max() + 1e-8)
        
        # Apply latitude weighting (sky more likely at top)
        lat_weight = np.zeros((H, W))
        for v in range(H):
            # Convert pixel row to latitude
            lat_norm = v / H  # 0 (top) to 1 (bottom)
            latitude = 90 - (lat_norm * 180)  # 90° to -90°
            
            # Weight based on latitude (higher weight for upper latitudes)
            if latitude > horizon_lat:
                weight = (latitude - horizon_lat) / (90 - horizon_lat)
                lat_weight[v, :] = weight
        
        # Combine gradient and latitude weighting
        sky_mask = sky_indicator * lat_weight
        
        return (sky_mask > threshold).astype(np.float32)
    
    def _blue_sky_detection(self, img: np.ndarray, threshold: float) -> np.ndarray:
        """Detect sky using blue color characteristics."""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Sky detection based on blue hues and high saturation
        hue = hsv[:, :, 0] / 179.0  # Normalize to [0, 1]
        sat = hsv[:, :, 1] / 255.0  # Normalize to [0, 1] 
        val = hsv[:, :, 2] / 255.0  # Normalize to [0, 1]
        
        # Blue sky characteristics: hue 0.5-0.7 (blue), moderate saturation, high value
        blue_mask = (
            ((hue > 0.45) & (hue < 0.75)) &  # Blue hues
            (sat > 0.2) &                    # Some saturation
            (val > 0.3)                      # Not too dark
        ).astype(np.float32)
        
        # Smooth the mask
        blue_mask = cv2.GaussianBlur(blue_mask, (5, 5), 1.0)
        
        return (blue_mask > threshold).astype(np.float32)
    
    def _horizon_line_detection(self, img: np.ndarray, horizon_lat: float, min_sky_height: float) -> np.ndarray:
        """Simple horizon line based sky detection."""
        H, W = img.shape[:2]
        
        # For mountain panorama, assume sky is in upper portion
        # Default to upper 30% if horizon_lat is 0
        if abs(horizon_lat) < 1e-6:  # Default horizon
            min_sky_height = max(min_sky_height, 0.3)  # At least 30% for mountains
        
        # Ensure minimum sky height (more generous for mountain scenes)
        min_sky_rows = int(min_sky_height * H)
        
        # Convert horizon latitude to pixel row
        lat_norm = (90 - horizon_lat) / 180.0  # Convert to [0, 1]
        horizon_row = int(lat_norm * H)
        
        # Use the more generous of the two
        actual_sky_rows = max(min_sky_rows, horizon_row)
        actual_sky_rows = min(actual_sky_rows, int(0.6 * H))  # Cap at 60% of image
        
        # Create mask
        sky_mask = np.zeros((H, W), dtype=np.float32)
        sky_mask[:actual_sky_rows, :] = 1.0
        
        log.info(f"Horizon detection: sky_rows={actual_sky_rows}, coverage={actual_sky_rows/H:.1%}")
        
        return sky_mask
    
    def _combine_with_hint(self, auto_mask: np.ndarray, hint_mask: np.ndarray, blend_factor: float) -> np.ndarray:
        """Combine automatic detection with manual hint."""
        # Normalize hint mask
        hint_norm = (hint_mask - hint_mask.min()) / (hint_mask.max() - hint_mask.min() + 1e-8)
        
        # Blend with automatic detection
        combined = auto_mask * (1 - blend_factor) + hint_norm * blend_factor
        
        return np.clip(combined, 0, 1)
    
    def _apply_feathering(self, mask: np.ndarray, feather_amount: float, height: int, width: int) -> np.ndarray:
        """Apply smooth feathering to mask edges."""
        # Convert feather amount to pixel radius
        feather_radius = int(feather_amount * min(height, width))
        
        if feather_radius > 0:
            # Use Gaussian blur for smooth falloff
            mask_blurred = cv2.GaussianBlur(mask.astype(np.float32), 
                                          (feather_radius * 2 + 1, feather_radius * 2 + 1), 
                                          feather_radius / 3.0)
            return mask_blurred
        
        return mask
    
    def _fix_erp_seam(self, mask: np.ndarray) -> np.ndarray:
        """Ensure mask continuity across ERP longitude seam (±180°)."""
        H, W = mask.shape
        
        # Get seam columns
        left_edge = mask[:, 0]     # -180°
        right_edge = mask[:, -1]   # +180° 
        
        # Average the seam values
        averaged_edge = (left_edge + right_edge) / 2
        
        # Apply to both edges
        mask[:, 0] = averaged_edge
        mask[:, -1] = averaged_edge
        
        return mask
    
    def _create_preview(self, img: np.ndarray, sky_mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
        """Create a visualization showing the layer separation."""
        H, W = img.shape[:2]
        
        # Create colored overlay
        preview = img.copy()
        
        # Tint sky regions blue
        sky_overlay = np.zeros_like(img)
        sky_overlay[:, :, 2] = 0.3  # Blue tint
        preview = preview * (1 - sky_mask[..., np.newaxis] * 0.3) + sky_overlay * sky_mask[..., np.newaxis]
        
        # Tint background regions green  
        bg_overlay = np.zeros_like(img)
        bg_overlay[:, :, 1] = 0.2  # Green tint
        preview = preview * (1 - bg_mask[..., np.newaxis] * 0.2) + bg_overlay * bg_mask[..., np.newaxis]
        
        return np.clip(preview, 0, 1).astype(np.float32)


class Layer_Builder_360:
    """Build structured layers from masks and ERP image with proper depth assignment."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "erp_image": ("IMAGE",),     # [1, H, W, 3]
                "erp_depth": ("IMAGE",),     # [1, H, W, 3] (depth in all channels)
                "sky_mask": ("IMAGE",),      # [1, H, W, 3] 
                "background_mask": ("IMAGE",), # [1, H, W, 3]
                "sky_depth_value": ("FLOAT", {"default": 1000.0, "min": 100.0, "max": 10000.0, "step": 10.0}),
                "depth_scale_bg": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "layer_priority": (["sky_back", "back_sky"], {"default": "sky_back"}),
                "edge_cleanup": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "object_masks": ("IMAGE",),   # Optional object masks [N, H, W, 3]
            }
        }

    RETURN_TYPES = ("LAYER_STACK",)
    RETURN_NAMES = ("layer_stack",)
    FUNCTION = "build_layers"
    CATEGORY = "MoGe360/Layers"
    DESCRIPTION = "Build structured layers from masks with proper depth assignment"

    def build_layers(self, erp_image: torch.Tensor, erp_depth: torch.Tensor, 
                    sky_mask: torch.Tensor, background_mask: torch.Tensor,
                    sky_depth_value: float, depth_scale_bg: float, layer_priority: str,
                    edge_cleanup: bool, object_masks: torch.Tensor = None):
        
        device = mm.get_torch_device()
        B, H, W, C = erp_image.shape
        
        # Convert inputs to numpy
        img_np = erp_image[0].cpu().numpy()  # [H, W, 3]
        depth_np = erp_depth[0, :, :, 0].cpu().numpy()  # [H, W]
        sky_np = sky_mask[0, :, :, 0].cpu().numpy()  # [H, W]
        bg_np = background_mask[0, :, :, 0].cpu().numpy()  # [H, W]
        
        log.info(f"Building layers from {W}x{H} ERP with {sky_np.sum():.0f} sky pixels")
        
        # Initialize layer list
        layers = []
        
        # Create sky layer
        sky_layer = self._create_sky_layer(img_np, depth_np, sky_np, sky_depth_value, H, W)
        layers.append(sky_layer)
        
        # Create background layer
        bg_layer = self._create_background_layer(img_np, depth_np, bg_np, depth_scale_bg, H, W)
        layers.append(bg_layer)
        
        # Add object layers if provided
        if object_masks is not None:
            obj_layers = self._create_object_layers(img_np, depth_np, object_masks, H, W)
            layers.extend(obj_layers)
        
        # Apply edge cleanup if requested
        if edge_cleanup:
            layers = self._cleanup_layer_edges(layers)
        
        # Ensure ERP seam continuity for all layers
        layers = self._fix_all_layer_seams(layers)
        
        # Sort layers by priority
        if layer_priority == "back_sky":
            # Background first, then sky (sky in front)
            layers = [l for l in layers if l['type'] == 'background'] + \
                    [l for l in layers if l['type'] == 'sky'] + \
                    [l for l in layers if l['type'] == 'object']
        
        # Package into layer stack
        layer_stack = {
            'layers': layers,
            'dimensions': (H, W),
            'layer_count': len(layers),
            'metadata': {
                'sky_depth': sky_depth_value,
                'bg_scale': depth_scale_bg,
                'priority': layer_priority,
                'has_objects': object_masks is not None
            }
        }
        
        log.info(f"Created {len(layers)} layers: {[l['type'] for l in layers]}")
        
        return (layer_stack,)
    
    def _create_sky_layer(self, img: np.ndarray, depth: np.ndarray, sky_mask: np.ndarray, 
                         sky_depth: float, H: int, W: int) -> Dict[str, Any]:
        """Create sky layer with infinite/far depth."""
        
        # Sky RGB from original image
        sky_rgb = img * sky_mask[..., np.newaxis]
        
        # Sky gets constant far depth
        sky_depth_map = np.full((H, W), sky_depth, dtype=np.float32) * sky_mask
        
        # Alpha is the sky mask itself
        sky_alpha = sky_mask.astype(np.float32)
        
        return {
            'type': 'sky',
            'rgb': sky_rgb,
            'alpha': sky_alpha, 
            'depth': sky_depth_map,
            'priority': 0,  # Background priority
            'metadata': {
                'pixel_count': int(sky_mask.sum()),
                'coverage': float(sky_mask.mean()),
                'depth_value': sky_depth
            }
        }
    
    def _create_background_layer(self, img: np.ndarray, depth: np.ndarray, bg_mask: np.ndarray,
                               depth_scale: float, H: int, W: int) -> Dict[str, Any]:
        """Create background layer with scaled MoGe depth."""
        
        # Background RGB from original image
        bg_rgb = img * bg_mask[..., np.newaxis]
        
        # Background uses scaled MoGe depth
        bg_depth_map = depth * depth_scale * bg_mask
        
        # Alpha is the background mask
        bg_alpha = bg_mask.astype(np.float32)
        
        return {
            'type': 'background',
            'rgb': bg_rgb,
            'alpha': bg_alpha,
            'depth': bg_depth_map,
            'priority': 1,  # Middle priority
            'metadata': {
                'pixel_count': int(bg_mask.sum()),
                'coverage': float(bg_mask.mean()),
                'depth_scale': depth_scale,
                'depth_range': (float(bg_depth_map[bg_mask > 0].min()) if bg_mask.sum() > 0 else 0,
                              float(bg_depth_map[bg_mask > 0].max()) if bg_mask.sum() > 0 else 0)
            }
        }
    
    def _create_object_layers(self, img: np.ndarray, depth: np.ndarray, 
                            object_masks: torch.Tensor, H: int, W: int) -> List[Dict[str, Any]]:
        """Create individual object layers from object masks."""
        
        obj_masks_np = object_masks.cpu().numpy()  # [N, H, W, 3]
        N = obj_masks_np.shape[0]
        
        object_layers = []
        
        for i in range(N):
            obj_mask = obj_masks_np[i, :, :, 0]  # [H, W]
            
            if obj_mask.sum() < 10:  # Skip tiny objects
                continue
            
            # Object RGB
            obj_rgb = img * obj_mask[..., np.newaxis]
            
            # Object depth (use MoGe depth as-is for objects)
            obj_depth_map = depth * obj_mask
            
            # Alpha is the object mask
            obj_alpha = obj_mask.astype(np.float32)
            
            object_layers.append({
                'type': 'object',
                'rgb': obj_rgb,
                'alpha': obj_alpha,
                'depth': obj_depth_map,
                'priority': 2 + i,  # Foreground priority
                'metadata': {
                    'object_id': i,
                    'pixel_count': int(obj_mask.sum()),
                    'coverage': float(obj_mask.mean()),
                    'depth_range': (float(obj_depth_map[obj_mask > 0].min()) if obj_mask.sum() > 0 else 0,
                                  float(obj_depth_map[obj_mask > 0].max()) if obj_mask.sum() > 0 else 0)
                }
            })
        
        return object_layers
    
    def _cleanup_layer_edges(self, layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean up layer edges to reduce artifacts."""
        
        for layer in layers:
            alpha = layer['alpha']
            
            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Close small holes
            alpha_cleaned = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
            
            # Remove small isolated regions
            alpha_cleaned = cv2.morphologyEx(alpha_cleaned, cv2.MORPH_OPEN, kernel)
            
            # Smooth edges slightly
            alpha_cleaned = cv2.GaussianBlur(alpha_cleaned, (3, 3), 0.5)
            
            # Update layer
            layer['alpha'] = alpha_cleaned
            layer['rgb'] = layer['rgb'] * alpha_cleaned[..., np.newaxis]
            layer['depth'] = layer['depth'] * alpha_cleaned
        
        return layers
    
    def _fix_all_layer_seams(self, layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix ERP seam continuity for all layers."""
        
        for layer in layers:
            # Fix RGB seam
            rgb = layer['rgb']
            rgb[:, 0] = (rgb[:, 0] + rgb[:, -1]) / 2
            rgb[:, -1] = rgb[:, 0]
            
            # Fix alpha seam  
            alpha = layer['alpha']
            alpha[:, 0] = (alpha[:, 0] + alpha[:, -1]) / 2
            alpha[:, -1] = alpha[:, 0]
            
            # Fix depth seam
            depth = layer['depth']
            depth[:, 0] = (depth[:, 0] + depth[:, -1]) / 2
            depth[:, -1] = depth[:, 0]
            
            # Update layer
            layer['rgb'] = rgb
            layer['alpha'] = alpha 
            layer['depth'] = depth
        
        return layers