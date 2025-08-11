"""
Inpainting nodes for completing layers with ERP-safe processing.
Fills occluded regions behind objects in 360° panoramas with proper seam continuity.
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Any, Optional
import logging

from .spherical_utils import SphericalProjection
import comfy.model_management as mm

log = logging.getLogger(__name__)

class Layer_Complete_360:
    """Complete layer RGB data by inpainting occluded regions with ERP-aware processing."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_stack": ("LAYER_STACK",),
                "inpaint_method": (["opencv_telea", "opencv_ns", "edge_extend", "patch_match"], {"default": "opencv_telea"}),
                "inpaint_radius": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "blend_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
                "seam_protection": ("BOOLEAN", {"default": True}),
                "quality_mode": (["fast", "balanced", "high"], {"default": "balanced"}),
            },
            "optional": {
                "custom_inpaint_mask": ("IMAGE",),  # Optional manual inpaint regions
                "background_reference": ("IMAGE",),  # Reference for inpainting context
            }
        }

    RETURN_TYPES = ("LAYER_STACK",)
    RETURN_NAMES = ("completed_layers",)
    FUNCTION = "complete_layers"
    CATEGORY = "MoGe360/Layers"
    DESCRIPTION = "Complete layers by inpainting occluded regions with ERP seam continuity"

    def complete_layers(self, layer_stack: Dict, inpaint_method: str, inpaint_radius: int, 
                       blend_strength: float, seam_protection: bool, quality_mode: str,
                       custom_inpaint_mask: torch.Tensor = None, 
                       background_reference: torch.Tensor = None):
        
        device = mm.get_torch_device()
        
        layers = layer_stack['layers']
        H, W = layer_stack['dimensions']
        
        log.info(f"Completing {len(layers)} layers using {inpaint_method} method (quality: {quality_mode})")
        
        # Process each layer
        completed_layers = []
        
        for i, layer in enumerate(layers):
            layer_type = layer['type']
            layer_rgb = layer['rgb']  # [H, W, 3]
            layer_alpha = layer['alpha']  # [H, W]
            
            log.info(f"Processing layer {i+1}: {layer_type} ({layer_alpha.sum():.0f} pixels)")
            
            # Determine what needs inpainting
            inpaint_mask = self._create_inpaint_mask(layer_alpha, layer_type, H, W)
            
            # Add custom inpaint regions if provided
            if custom_inpaint_mask is not None:
                custom_mask = custom_inpaint_mask[0, :, :, 0].cpu().numpy()
                inpaint_mask = np.maximum(inpaint_mask, custom_mask)
            
            if inpaint_mask.sum() > 0:
                log.info(f"Inpainting {inpaint_mask.sum():.0f} pixels in {layer_type} layer")
                
                # Perform ERP-safe inpainting
                completed_rgb = self._inpaint_layer_erp_safe(
                    layer_rgb, inpaint_mask, inpaint_method, inpaint_radius, 
                    quality_mode, seam_protection, background_reference
                )
                
                # Blend completed regions with original
                final_rgb = layer_rgb * (1 - inpaint_mask[..., np.newaxis]) + \
                          completed_rgb * inpaint_mask[..., np.newaxis] * blend_strength + \
                          layer_rgb * inpaint_mask[..., np.newaxis] * (1 - blend_strength)
                
                # Update layer
                completed_layer = layer.copy()
                completed_layer['rgb'] = final_rgb
                completed_layer['metadata']['inpainted_pixels'] = int(inpaint_mask.sum())
                completed_layer['metadata']['inpaint_method'] = inpaint_method
                
            else:
                log.info(f"No inpainting needed for {layer_type} layer")
                completed_layer = layer.copy()
                completed_layer['metadata']['inpainted_pixels'] = 0
            
            completed_layers.append(completed_layer)
        
        # Update layer stack
        completed_stack = layer_stack.copy()
        completed_stack['layers'] = completed_layers
        completed_stack['metadata']['inpainting_applied'] = True
        completed_stack['metadata']['inpaint_method'] = inpaint_method
        
        total_inpainted = sum(l['metadata'].get('inpainted_pixels', 0) for l in completed_layers)
        log.info(f"Completed {len(completed_layers)} layers, inpainted {total_inpainted} total pixels")
        
        return (completed_stack,)
    
    def _create_inpaint_mask(self, alpha: np.ndarray, layer_type: str, H: int, W: int) -> np.ndarray:
        """Create mask for regions that need inpainting."""
        
        # Start with inverted alpha (holes that need filling)
        inpaint_mask = (1.0 - alpha).astype(np.float32)
        
        # Different strategies based on layer type
        if layer_type == "sky":
            # Sky layers: fill small holes, extend edges
            inpaint_mask = self._refine_sky_inpaint_mask(inpaint_mask, alpha)
            
        elif layer_type == "background":
            # Background: fill holes where objects were removed
            inpaint_mask = self._refine_background_inpaint_mask(inpaint_mask, alpha)
            
        elif layer_type == "object":
            # Objects: minimal inpainting, mostly edge cleanup
            inpaint_mask = self._refine_object_inpaint_mask(inpaint_mask, alpha)
        
        # Ensure seam continuity
        inpaint_mask = self._fix_erp_seam(inpaint_mask)
        
        return (inpaint_mask > 0.1).astype(np.float32)
    
    def _refine_sky_inpaint_mask(self, mask: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Refine inpainting mask for sky layer."""
        
        # Close small holes in sky
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Extend sky boundaries slightly to avoid edge artifacts
        sky_boundary = cv2.dilate(alpha, kernel, iterations=2) - alpha
        mask = np.maximum(mask, sky_boundary * 0.5)
        
        return mask
    
    def _refine_background_inpaint_mask(self, mask: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Refine inpainting mask for background layer."""
        
        # Fill larger holes where objects were removed
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Expand mask slightly to ensure good inpainting context
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _refine_object_inpaint_mask(self, mask: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Refine inpainting mask for object layer."""
        
        # Minimal inpainting for objects - only fill tiny holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Only keep small holes
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Only fill small holes
                cv2.fillPoly(filtered_mask, [contour], 1.0)
        
        return filtered_mask
    
    def _inpaint_layer_erp_safe(self, rgb: np.ndarray, mask: np.ndarray, method: str, 
                               radius: int, quality: str, seam_protection: bool,
                               bg_reference: torch.Tensor = None) -> np.ndarray:
        """Perform ERP-safe inpainting with seam continuity."""
        
        H, W, C = rgb.shape
        
        # Convert to uint8 for OpenCV
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        if seam_protection:
            # Use circular padding to ensure seam continuity
            return self._inpaint_with_circular_padding(rgb_uint8, mask_uint8, method, radius, quality)
        else:
            # Direct inpainting
            return self._inpaint_direct(rgb_uint8, mask_uint8, method, radius, quality)
    
    def _inpaint_with_circular_padding(self, rgb: np.ndarray, mask: np.ndarray, 
                                     method: str, radius: int, quality: str) -> np.ndarray:
        """Inpaint with circular padding to ensure ERP seam continuity."""
        
        H, W = rgb.shape[:2]
        
        # Add circular padding
        pad_width = W // 4  # 25% padding
        
        # Create padded image and mask
        left_pad_img = rgb[:, -pad_width:]
        right_pad_img = rgb[:, :pad_width]
        padded_img = np.concatenate([left_pad_img, rgb, right_pad_img], axis=1)
        
        left_pad_mask = mask[:, -pad_width:]
        right_pad_mask = mask[:, :pad_width]
        padded_mask = np.concatenate([left_pad_mask, mask, right_pad_mask], axis=1)
        
        # Inpaint on padded image
        inpainted_padded = self._inpaint_direct(padded_img, padded_mask, method, radius, quality)
        
        # Extract center region (original image size)
        inpainted_result = inpainted_padded[:, pad_width:-pad_width]
        
        # Ensure result has correct dimensions
        assert inpainted_result.shape[:2] == (H, W), f"Size mismatch: {inpainted_result.shape[:2]} vs {(H, W)}"
        
        # Convert back to float32
        return (inpainted_result / 255.0).astype(np.float32)
    
    def _inpaint_direct(self, rgb: np.ndarray, mask: np.ndarray, method: str, 
                       radius: int, quality: str) -> np.ndarray:
        """Direct inpainting using specified method."""
        
        try:
            if method == "opencv_telea":
                result = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
                
            elif method == "opencv_ns":
                result = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_NS)
                
            elif method == "edge_extend":
                result = self._edge_extend_inpaint(rgb, mask, radius)
                
            elif method == "patch_match":
                result = self._patch_match_inpaint(rgb, mask, radius, quality)
                
            else:
                log.warning(f"Unknown inpaint method {method}, using Telea")
                result = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
            
            return result
            
        except Exception as e:
            log.error(f"Inpainting failed: {e}, returning original image")
            return rgb
    
    def _edge_extend_inpaint(self, rgb: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
        """Simple edge extension inpainting."""
        
        result = rgb.copy()
        mask_bool = mask > 127
        
        # Iteratively extend edges inward
        for iteration in range(radius):
            # Find edge pixels (have valid neighbors)
            kernel = np.ones((3, 3), np.uint8)
            dilated_valid = cv2.dilate((~mask_bool).astype(np.uint8), kernel)
            edge_pixels = mask_bool & (dilated_valid > 0)
            
            if not edge_pixels.any():
                break
            
            # For each edge pixel, take average of valid neighbors
            for y, x in np.column_stack(np.where(edge_pixels)):
                neighbor_sum = np.zeros(3, dtype=np.float32)
                neighbor_count = 0
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < rgb.shape[0] and 0 <= nx < rgb.shape[1] and 
                            not mask_bool[ny, nx]):
                            neighbor_sum += result[ny, nx].astype(np.float32)
                            neighbor_count += 1
                
                if neighbor_count > 0:
                    result[y, x] = (neighbor_sum / neighbor_count).astype(np.uint8)
                    mask_bool[y, x] = False
        
        return result
    
    def _patch_match_inpaint(self, rgb: np.ndarray, mask: np.ndarray, radius: int, quality: str) -> np.ndarray:
        """Patch-based inpainting (simplified implementation)."""
        
        # For now, fallback to Telea with larger radius for better quality
        adjusted_radius = radius * 2 if quality == "high" else radius
        
        try:
            return cv2.inpaint(rgb, mask, adjusted_radius, cv2.INPAINT_TELEA)
        except:
            return self._edge_extend_inpaint(rgb, mask, radius)
    
    def _fix_erp_seam(self, mask: np.ndarray) -> np.ndarray:
        """Ensure mask continuity across ERP longitude seam."""
        H, W = mask.shape
        
        # Get seam columns
        left_edge = mask[:, 0]
        right_edge = mask[:, -1]
        
        # Average seam values for continuity
        averaged_edge = (left_edge + right_edge) / 2
        
        # Apply to both edges
        mask[:, 0] = averaged_edge
        mask[:, -1] = averaged_edge
        
        return mask


class Layer_Alpha_Refiner:
    """Refine layer alpha masks with ERP-aware smoothing and edge detection."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_stack": ("LAYER_STACK",),
                "smoothing_method": (["gaussian", "bilateral", "edge_preserving"], {"default": "bilateral"}),
                "smoothing_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "edge_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cleanup_small_regions": ("BOOLEAN", {"default": True}),
                "min_region_size": ("INT", {"default": 50, "min": 10, "max": 500, "step": 10}),
            }
        }

    RETURN_TYPES = ("LAYER_STACK",)
    RETURN_NAMES = ("refined_layers",)
    FUNCTION = "refine_alphas"
    CATEGORY = "MoGe360/Layers"
    DESCRIPTION = "Refine layer alpha masks with ERP-aware smoothing"

    def refine_alphas(self, layer_stack: Dict, smoothing_method: str, smoothing_strength: float,
                     edge_threshold: float, cleanup_small_regions: bool, min_region_size: int):
        
        layers = layer_stack['layers']
        H, W = layer_stack['dimensions']
        
        log.info(f"Refining alpha masks for {len(layers)} layers using {smoothing_method}")
        
        refined_layers = []
        
        for layer in layers:
            original_alpha = layer['alpha']
            layer_type = layer['type']
            
            # Apply smoothing
            if smoothing_method == "gaussian":
                refined_alpha = self._gaussian_smooth(original_alpha, smoothing_strength)
            elif smoothing_method == "bilateral":
                refined_alpha = self._bilateral_smooth(original_alpha, layer['rgb'], smoothing_strength)
            else:  # edge_preserving
                refined_alpha = self._edge_preserving_smooth(original_alpha, layer['rgb'], smoothing_strength)
            
            # Clean up small regions
            if cleanup_small_regions:
                refined_alpha = self._cleanup_small_regions(refined_alpha, min_region_size)
            
            # Ensure ERP seam continuity
            refined_alpha = self._fix_erp_seam(refined_alpha)
            
            # Update layer
            refined_layer = layer.copy()
            refined_layer['alpha'] = refined_alpha
            refined_layer['rgb'] = layer['rgb'] * refined_alpha[..., np.newaxis]  # Update RGB masking
            refined_layer['metadata']['alpha_refined'] = True
            
            refined_layers.append(refined_layer)
        
        # Update layer stack
        refined_stack = layer_stack.copy()
        refined_stack['layers'] = refined_layers
        refined_stack['metadata']['alpha_refinement_applied'] = True
        
        log.info(f"Refined alpha masks for {len(refined_layers)} layers")
        
        return (refined_stack,)
    
    def _gaussian_smooth(self, alpha: np.ndarray, strength: float) -> np.ndarray:
        """Apply Gaussian smoothing to alpha mask."""
        kernel_size = int(strength * 2) * 2 + 1  # Ensure odd
        return cv2.GaussianBlur(alpha, (kernel_size, kernel_size), strength)
    
    def _bilateral_smooth(self, alpha: np.ndarray, rgb: np.ndarray, strength: float) -> np.ndarray:
        """Apply bilateral filtering for edge-preserving smoothing."""
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        # Use RGB as guide for bilateral filtering
        rgb_gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter
        diameter = int(strength * 2) + 1
        sigma_color = strength * 50
        sigma_space = strength * 50
        
        smoothed = cv2.bilateralFilter(alpha_uint8, diameter, sigma_color, sigma_space)
        
        return (smoothed / 255.0).astype(np.float32)
    
    def _edge_preserving_smooth(self, alpha: np.ndarray, rgb: np.ndarray, strength: float) -> np.ndarray:
        """Edge-preserving smoothing based on RGB gradients."""
        alpha_uint8 = (alpha * 255).astype(np.uint8)
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        
        # Detect edges in RGB
        rgb_gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(rgb_gray, 50, 150)
        edge_mask = (edges > 0).astype(np.float32)
        
        # Apply stronger smoothing away from edges
        strong_smooth = cv2.GaussianBlur(alpha, (int(strength*4)+1, int(strength*4)+1), strength*2)
        weak_smooth = cv2.GaussianBlur(alpha, (int(strength)+1, int(strength)+1), strength*0.5)
        
        # Blend based on edge strength
        result = alpha * edge_mask + strong_smooth * (1 - edge_mask) * 0.7 + weak_smooth * (1 - edge_mask) * 0.3
        
        return np.clip(result, 0, 1).astype(np.float32)
    
    def _cleanup_small_regions(self, alpha: np.ndarray, min_size: int) -> np.ndarray:
        """Remove small disconnected regions from alpha mask."""
        
        # Convert to binary
        binary = (alpha > 0.5).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Create cleaned mask
        cleaned = np.zeros_like(alpha)
        
        for label in range(1, num_labels):  # Skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_size:
                # Keep this region
                region_mask = (labels == label).astype(np.float32)
                # Apply original alpha values within this region
                cleaned += alpha * region_mask
        
        return cleaned
    
    def _fix_erp_seam(self, alpha: np.ndarray) -> np.ndarray:
        """Ensure alpha continuity across ERP seam."""
        H, W = alpha.shape
        
        # Average seam edges
        left_edge = alpha[:, 0]
        right_edge = alpha[:, -1]
        averaged_edge = (left_edge + right_edge) / 2
        
        alpha[:, 0] = averaged_edge
        alpha[:, -1] = averaged_edge
        
        return alpha