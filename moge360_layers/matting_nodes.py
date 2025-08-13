"""
Matting nodes for converting detection boxes to high-quality masks.
Implements ZIM matting and other mask refinement techniques for 360° panoramas.
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

class Boxes_To_ZIM_Mattes:
    """Convert detection boxes to high-quality masks using ZIM matting or fallback methods."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "erp_image": ("IMAGE", {
                    "tooltip": "Input equirectangular panorama image matching the detection input. Used as source for mask generation and refinement algorithms."
                }),
                "detection_boxes": ("DETECTION_BOXES", {
                    "tooltip": "Detection results from OWLViT_Detect_360 containing bounding boxes, confidence scores, and labels. Each box will be converted to a high-quality mask."
                }),
                "matting_method": (["zim", "grabcut", "simple_mask"], {
                    "default": "grabcut",
                    "tooltip": "Mask generation method: 'grabcut' uses OpenCV for intelligent segmentation, 'simple_mask' creates basic rectangular masks, 'zim' planned for future ZIM matting."
                }),
                "mask_expansion": ("FLOAT", {
                    "default": 1.2, "min": 1.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Expand bounding boxes before mask generation. 1.2 = 20% larger. Helps capture complete objects that detection boxes might not fully encompass."
                }),
                "edge_feather": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Smooth feathering amount for mask edges as percentage of image size. Higher values create softer transitions but may blur fine details. 0 = hard edges."
                }),
                "min_mask_size": ("INT", {
                    "default": 100, "min": 10, "max": 1000, "step": 10,
                    "tooltip": "Minimum pixel count for valid masks. Smaller masks are filtered out to remove noise detections. Adjust based on image resolution and object sizes."
                }),
            },
            "optional": {
                "trimap_erosion": ("INT", {
                    "default": 5, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Erosion iterations for trimap generation (future ZIM support). Controls how much to shrink the foreground region for better matting accuracy."
                }),
                "trimap_dilation": ("INT", {
                    "default": 10, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Dilation iterations for trimap generation (future ZIM support). Controls how much to expand the uncertain region for better edge refinement."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("object_masks",)
    OUTPUT_TOOLTIPS = (
        "Stack of object masks corresponding to detected objects. Each mask is a grayscale image where white areas indicate the object and black areas are background. ERP seam-safe.",
    )
    FUNCTION = "create_masks"
    CATEGORY = "MoGe360/Detection"
    DESCRIPTION = "Convert detection bounding boxes to high-quality object masks using GrabCut segmentation or simple expansion. Handles ERP seam continuity and provides mask refinement options."

    def create_masks(self, erp_image: torch.Tensor, detection_boxes: Dict, matting_method: str,
                    mask_expansion: float, edge_feather: float, min_mask_size: int,
                    trimap_erosion: int = 5, trimap_dilation: int = 10):
        
        device = mm.get_torch_device()
        B, H, W, C = erp_image.shape
        
        # Convert to numpy for processing
        img_np = erp_image[0].cpu().numpy()  # [H, W, 3]
        
        # Extract detection info
        boxes = detection_boxes.get('boxes', [])
        img_dimensions = detection_boxes.get('image_dimensions', (H, W))
        detection_count = detection_boxes.get('detection_count', 0)
        
        log.info(f"Creating {detection_count} masks using {matting_method} method")
        
        if not boxes:
            # Return empty mask stack
            empty_masks = torch.zeros((1, H, W, 3), device=device)
            return (empty_masks,)
        
        # Create masks for each detection
        all_masks = []
        
        for i, detection in enumerate(boxes):
            box = detection.get('box', [0, 0, 0, 0])
            confidence = detection.get('confidence', 0.0)
            label = detection.get('label', 'object')
            
            log.info(f"Processing detection {i+1}: {label} ({confidence:.2f}) at {box}")
            
            # Create mask for this detection
            if matting_method == "zim":
                mask = self._create_zim_mask(img_np, box, trimap_erosion, trimap_dilation)
            elif matting_method == "grabcut":
                mask = self._create_grabcut_mask(img_np, box, mask_expansion)
            else:  # simple_mask
                mask = self._create_simple_mask(img_dimensions, box, mask_expansion)
            
            # Apply size filter
            if mask is not None and np.sum(mask > 0.5) >= min_mask_size:
                # Apply edge feathering
                if edge_feather > 0:
                    mask = self._apply_feathering(mask, edge_feather)
                
                # Ensure ERP seam continuity
                mask = self._fix_erp_seam(mask)
                
                all_masks.append(mask)
                log.info(f"Created mask for {label}: {np.sum(mask > 0.5)} pixels")
            else:
                log.warning(f"Skipped small detection {i+1}: {label}")
        
        if not all_masks:
            # No valid masks created
            log.warning("No valid masks created from detections")
            empty_masks = torch.zeros((1, H, W, 3), device=device)
            return (empty_masks,)
        
        # Stack masks into tensor [N, H, W, 3]
        stacked_masks = np.stack(all_masks, axis=0)  # [N, H, W]
        
        # Convert to 3-channel masks
        mask_tensor = torch.from_numpy(stacked_masks).unsqueeze(-1).repeat(1, 1, 1, 3).to(device).float()
        
        log.info(f"Created {len(all_masks)} object masks: {mask_tensor.shape}")
        
        return (mask_tensor,)
    
    def _create_zim_mask(self, img: np.ndarray, box: List[float], 
                        trimap_erosion: int, trimap_dilation: int) -> Optional[np.ndarray]:
        """Create mask using ZIM matting (placeholder for now - would need ZIM ONNX model)."""
        log.info("ZIM matting not yet implemented - falling back to GrabCut")
        return self._create_grabcut_mask(img, box, 1.2)
    
    def _create_grabcut_mask(self, img: np.ndarray, box: List[float], 
                           expansion: float) -> Optional[np.ndarray]:
        """Create mask using OpenCV GrabCut algorithm."""
        try:
            H, W = img.shape[:2]
            x1, y1, x2, y2 = box
            
            # Expand box slightly
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            exp_w, exp_h = w * expansion, h * expansion
            x1_exp = max(0, cx - exp_w / 2)
            y1_exp = max(0, cy - exp_h / 2)
            x2_exp = min(W, cx + exp_w / 2)
            y2_exp = min(H, cy + exp_h / 2)
            
            # Convert to integer coordinates
            rect = (int(x1_exp), int(y1_exp), int(x2_exp - x1_exp), int(y2_exp - y1_exp))
            
            # Convert image to uint8 if needed
            if img.dtype != np.uint8:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img
            
            # Initialize mask
            mask = np.zeros((H, W), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Run GrabCut
            cv2.grabCut(img_uint8, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Extract foreground
            mask_out = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float32)
            
            return mask_out
            
        except Exception as e:
            log.warning(f"GrabCut failed: {e}, falling back to simple mask")
            return self._create_simple_mask((H, W), box, expansion)
    
    def _create_simple_mask(self, img_dims: Tuple[int, int], box: List[float], 
                          expansion: float) -> np.ndarray:
        """Create simple rectangular mask from bounding box."""
        H, W = img_dims
        x1, y1, x2, y2 = box
        
        # Expand box
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        exp_w, exp_h = w * expansion, h * expansion
        x1_exp = max(0, cx - exp_w / 2)
        y1_exp = max(0, cy - exp_h / 2)
        x2_exp = min(W, cx + exp_w / 2)
        y2_exp = min(H, cy + exp_h / 2)
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.float32)
        mask[int(y1_exp):int(y2_exp), int(x1_exp):int(x2_exp)] = 1.0
        
        # Add some smoothing to avoid hard edges
        kernel_size = max(3, int(min(exp_w, exp_h) / 20))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), kernel_size / 3)
        
        return mask
    
    def _apply_feathering(self, mask: np.ndarray, feather_amount: float) -> np.ndarray:
        """Apply smooth feathering to mask edges."""
        if feather_amount <= 0:
            return mask
        
        # Convert feather amount to pixel radius
        H, W = mask.shape[:2]
        feather_radius = int(feather_amount * min(H, W) / 100)  # Percentage of image size
        
        if feather_radius > 0:
            # Use distance transform for smooth falloff
            mask_binary = (mask > 0.5).astype(np.uint8)
            
            # Distance from edge
            dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
            
            # Create smooth falloff
            smooth_mask = np.clip(dist_transform / feather_radius, 0, 1)
            
            # Combine with original mask
            feathered_mask = mask * smooth_mask
            
            return feathered_mask.astype(np.float32)
        
        return mask
    
    def _fix_erp_seam(self, mask: np.ndarray) -> np.ndarray:
        """Ensure mask continuity across ERP longitude seam (±180°)."""
        H, W = mask.shape
        
        # Get seam columns
        left_edge = mask[:, 0]     # -180°
        right_edge = mask[:, -1]   # +180°
        
        # Average the seam values for continuity
        averaged_edge = (left_edge + right_edge) / 2
        
        # Apply to both edges
        mask[:, 0] = averaged_edge
        mask[:, -1] = averaged_edge
        
        return mask


class Detection_Mask_Combiner:
    """Combine multiple object masks with sky/background masks for complete layering."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "object_masks": ("IMAGE", {
                    "tooltip": "Stack of object masks from Boxes_To_ZIM_Mattes. These masks will be combined and used to modify sky and background masks for proper layer separation."
                }),
                "sky_mask": ("IMAGE", {
                    "tooltip": "Sky mask from Sky_Background_Splitter. Object areas will be removed from this mask to prevent sky/object overlap in the final layering."
                }),
                "background_mask": ("IMAGE", {
                    "tooltip": "Background mask from Sky_Background_Splitter. Object areas will be removed from this mask to create clean background layer without objects."
                }),
                "combination_method": (["priority", "alpha_blend", "max_blend"], {
                    "default": "priority",
                    "tooltip": "Method for combining masks: 'priority' gives objects precedence, 'alpha_blend' creates soft transitions, 'max_blend' preserves maximum values."
                }),
                "object_priority": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Strength of object mask influence when removing from sky/background. 1.0 = full removal, 0.5 = partial removal, 2.0 = extended removal area."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("combined_sky", "combined_background", "combined_objects")
    OUTPUT_TOOLTIPS = (
        "Sky mask with object areas removed. Clean sky layer without object contamination, ready for separate sky processing and depth assignment.",
        "Background mask with object areas removed. Clean background layer containing terrain and static elements, excluding sky and detected objects.",
        "Combined object masks from all detections. Single mask containing all detected objects, useful for layer building and depth processing."
    )
    FUNCTION = "combine_masks"
    CATEGORY = "MoGe360/Detection"
    DESCRIPTION = "Combine multiple object masks with sky and background masks to create clean, non-overlapping layer masks. Ensures proper layer separation for 3D reconstruction."

    def combine_masks(self, object_masks: torch.Tensor, sky_mask: torch.Tensor, 
                     background_mask: torch.Tensor, combination_method: str, object_priority: float):
        
        device = mm.get_torch_device()
        
        # Get dimensions
        N, H, W, C = object_masks.shape  # N object masks
        sky_np = sky_mask[0, :, :, 0].cpu().numpy()  # [H, W]
        bg_np = background_mask[0, :, :, 0].cpu().numpy()  # [H, W]
        
        log.info(f"Combining {N} object masks with sky/background masks using {combination_method}")
        
        # Combine all object masks into single mask
        combined_objects = torch.max(object_masks, dim=0)[0]  # [H, W, 3]
        combined_obj_np = combined_objects[:, :, 0].cpu().numpy()  # [H, W]
        
        if combination_method == "priority":
            # Objects have priority over sky/background
            # Remove object areas from sky and background
            final_sky = sky_np * (1 - combined_obj_np * object_priority)
            final_bg = bg_np * (1 - combined_obj_np * object_priority)
            final_obj = combined_obj_np
            
        elif combination_method == "alpha_blend":
            # Soft blending
            alpha = combined_obj_np * object_priority
            final_sky = sky_np * (1 - alpha)
            final_bg = bg_np * (1 - alpha)
            final_obj = combined_obj_np
            
        else:  # max_blend
            # Take maximum values
            final_sky = np.maximum(sky_np - combined_obj_np * object_priority, 0)
            final_bg = np.maximum(bg_np - combined_obj_np * object_priority, 0)
            final_obj = combined_obj_np
        
        # Convert back to tensors
        sky_tensor = torch.from_numpy(final_sky).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        bg_tensor = torch.from_numpy(final_bg).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        obj_tensor = torch.from_numpy(final_obj).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
        
        log.info(f"Combined masks - Sky: {final_sky.sum():.0f}px, BG: {final_bg.sum():.0f}px, Obj: {final_obj.sum():.0f}px")
        
        return (sky_tensor, bg_tensor, obj_tensor)