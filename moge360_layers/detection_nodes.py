"""
Object detection nodes for 360° panorama processing.
Implements OWL-ViT open-vocabulary detection with ERP-aware processing.
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

class OWLViT_Detect_360:
    """Open-vocabulary object detection for equirectangular panoramas using OWL-ViT."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "erp_image": ("IMAGE",),  # [1, H, W, 3]
                "text_queries": ("STRING", {
                    "default": "mountain peak, rock formation, tree, building, person",
                    "multiline": True
                }),
                "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.1, "max": 0.9, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.01}),
                "erp_mode": (["circular_padding", "perspective_tiles", "direct"], {"default": "circular_padding"}),
                "max_detections": ("INT", {"default": 50, "min": 5, "max": 200, "step": 5}),
                "model_size": (["base", "large"], {"default": "base"}),
            },
            "optional": {
                "detection_hint_mask": ("IMAGE",),  # Optional region hint
            }
        }

    RETURN_TYPES = ("DETECTION_BOXES", "IMAGE", "STRING")
    RETURN_NAMES = ("detection_boxes", "detection_preview", "detection_summary")
    FUNCTION = "detect_objects"
    CATEGORY = "MoGe360/Detection"
    DESCRIPTION = "Detect objects in ERP panorama using OWL-ViT open-vocabulary detection"

    def detect_objects(self, erp_image: torch.Tensor, text_queries: str, confidence_threshold: float,
                      nms_threshold: float, erp_mode: str, max_detections: int, model_size: str,
                      detection_hint_mask: torch.Tensor = None):
        
        device = mm.get_torch_device()
        B, H, W, C = erp_image.shape
        
        # Convert to numpy for processing
        img_np = erp_image[0].cpu().numpy()  # [H, W, 3]
        
        # Parse text queries
        queries = [q.strip() for q in text_queries.split(',') if q.strip()]
        if not queries:
            queries = ["object"]
        
        log.info(f"OWL-ViT detection on {W}x{H} ERP with {len(queries)} queries: {queries[:3]}...")
        
        # Load OWL-ViT model
        model, processor = self._load_owlvit_model(model_size, device)
        
        # Apply ERP-aware detection based on mode
        if erp_mode == "circular_padding":
            detections = self._detect_with_circular_padding(
                img_np, model, processor, queries, confidence_threshold, nms_threshold, max_detections
            )
        elif erp_mode == "perspective_tiles":
            detections = self._detect_with_perspective_tiles(
                img_np, model, processor, queries, confidence_threshold, nms_threshold, max_detections, H, W
            )
        else:  # direct
            detections = self._detect_direct(
                img_np, model, processor, queries, confidence_threshold, nms_threshold, max_detections
            )
        
        # Filter with hint mask if provided
        if detection_hint_mask is not None:
            hint_np = detection_hint_mask[0, :, :, 0].cpu().numpy()
            detections = self._filter_with_hint(detections, hint_np, H, W)
        
        # Ensure ERP seam continuity
        detections = self._fix_erp_seam_detections(detections, W)
        
        # Create detection preview
        preview_img = self._create_detection_preview(img_np, detections)
        
        # Create summary
        summary = self._create_detection_summary(detections, queries)
        
        # Package detections
        detection_boxes = {
            'boxes': detections,
            'image_dimensions': (H, W),
            'queries': queries,
            'confidence_threshold': confidence_threshold,
            'detection_count': len(detections)
        }
        
        # Convert preview to tensor
        preview_tensor = torch.from_numpy(preview_img).unsqueeze(0).to(device)
        
        log.info(f"Detected {len(detections)} objects above confidence {confidence_threshold}")
        
        return (detection_boxes, preview_tensor, summary)
    
    def _load_owlvit_model(self, model_size: str, device):
        """Load OWL-ViT model and processor."""
        try:
            from transformers import OwlViTModel, OwlViTProcessor
            
            model_name = f"google/owlv2-{model_size}-patch16-ensemble"
            
            log.info(f"Loading OWL-ViT model: {model_name}")
            
            # Load model
            processor = OwlViTProcessor.from_pretrained(model_name)
            model = OwlViTModel.from_pretrained(model_name)
            model.to(device)
            model.eval()
            
            return model, processor
            
        except ImportError:
            raise ImportError("transformers library not found. Please install: pip install transformers")
        except Exception as e:
            raise RuntimeError(f"Failed to load OWL-ViT model: {e}")
    
    def _detect_with_circular_padding(self, img: np.ndarray, model, processor, queries: List[str],
                                    confidence_threshold: float, nms_threshold: float, max_detections: int) -> List[Dict]:
        """Detect objects using circular padding to handle ERP seams."""
        
        H, W = img.shape[:2]
        
        # Add circular padding (duplicate left/right edges)
        padding_width = W // 4  # 25% padding on each side
        
        # Create padded image
        left_pad = img[:, -padding_width:]  # Right edge -> left padding
        right_pad = img[:, :padding_width]  # Left edge -> right padding
        padded_img = np.concatenate([left_pad, img, right_pad], axis=1)
        
        log.info(f"Created padded image: {padded_img.shape} (padding: {padding_width}px)")
        
        # Run detection on padded image
        padded_detections = self._detect_direct(
            padded_img, model, processor, queries, confidence_threshold, nms_threshold, max_detections
        )
        
        # Adjust detection coordinates back to original image
        detections = []
        padded_width = padded_img.shape[1]
        
        for det in padded_detections:
            # Adjust box coordinates
            x1, y1, x2, y2 = det['box']
            
            # Convert from padded coordinates to original coordinates
            x1_orig = x1 - padding_width
            x2_orig = x2 - padding_width
            
            # Handle boxes that span the seam
            if x1_orig < 0 and x2_orig < 0:
                # Box entirely in left padding - wrap to right side
                det['box'] = [x1_orig + W, y1, x2_orig + W, y2]
                detections.append(det)
            elif x1_orig >= W and x2_orig >= W:
                # Box entirely in right padding - wrap to left side  
                det['box'] = [x1_orig - W, y1, x2_orig - W, y2]
                detections.append(det)
            elif x1_orig < 0 < x2_orig or x1_orig < W < x2_orig:
                # Box spans seam - create two boxes
                if x1_orig < 0:
                    # Split box at seam
                    detections.append({
                        **det,
                        'box': [x1_orig + W, y1, W, y2],
                        'seam_box': True
                    })
                    detections.append({
                        **det, 
                        'box': [0, y1, x2_orig, y2],
                        'seam_box': True
                    })
                else:
                    # Box spans right edge
                    detections.append({
                        **det,
                        'box': [x1_orig, y1, W, y2],
                        'seam_box': True
                    })
                    detections.append({
                        **det,
                        'box': [0, y1, x2_orig - W, y2], 
                        'seam_box': True
                    })
            elif 0 <= x1_orig < W and 0 <= x2_orig < W:
                # Box in original image area
                det['box'] = [x1_orig, y1, x2_orig, y2]
                detections.append(det)
        
        return detections
    
    def _detect_with_perspective_tiles(self, img: np.ndarray, model, processor, queries: List[str],
                                     confidence_threshold: float, nms_threshold: float, max_detections: int,
                                     H: int, W: int) -> List[Dict]:
        """Detect objects by processing perspective tiles."""
        
        # Create tile grid (smaller than MoGe tiles for faster detection)
        tiles_params = SphericalProjection.create_tile_grid(
            grid_yaw=4, grid_pitch=2, fov_deg=90, overlap_deg=20
        )
        
        log.info(f"Processing {len(tiles_params)} perspective tiles for detection")
        
        all_detections = []
        tile_size = 512  # Smaller for faster detection
        
        for i, tile_params in enumerate(tiles_params):
            # Sample perspective tile
            tile_img, intrinsics, extrinsics = SphericalProjection.sample_perspective_tile(
                img,
                tile_params['center_lon'],
                tile_params['center_lat'], 
                tile_params['fov_deg'],
                tile_size
            )
            
            # Detect in this tile
            tile_detections = self._detect_direct(
                tile_img, model, processor, queries, confidence_threshold, nms_threshold, max_detections // len(tiles_params)
            )
            
            # Project detections back to ERP coordinates
            for det in tile_detections:
                erp_detection = self._project_detection_to_erp(
                    det, tile_params, intrinsics, tile_size, H, W
                )
                if erp_detection:
                    all_detections.append(erp_detection)
        
        # Remove duplicate detections across tiles
        all_detections = self._remove_duplicate_detections(all_detections, nms_threshold)
        
        return all_detections[:max_detections]
    
    def _detect_direct(self, img: np.ndarray, model, processor, queries: List[str],
                      confidence_threshold: float, nms_threshold: float, max_detections: int) -> List[Dict]:
        """Direct detection on image without ERP handling."""
        
        # Convert to PIL Image
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        
        # Prepare inputs
        inputs = processor(text=queries, images=pil_img, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process outputs
        target_sizes = torch.Tensor([pil_img.size[::-1]]).to(model.device)  # [H, W]
        results = processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )[0]  # First (and only) image
        
        # Convert to our format
        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy() 
        labels = results["labels"].cpu().numpy()
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score >= confidence_threshold and len(detections) < max_detections:
                x1, y1, x2, y2 = box
                detections.append({
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                    'label': queries[label],
                    'label_id': int(label)
                })
        
        # Apply NMS
        if len(detections) > 1:
            detections = self._apply_nms(detections, nms_threshold)
        
        return detections
    
    def _apply_nms(self, detections: List[Dict], threshold: float) -> List[Dict]:
        """Apply Non-Maximum Suppression."""
        if len(detections) <= 1:
            return detections
        
        # Convert to format for NMS
        boxes = np.array([det['box'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])
        
        # Convert to x1, y1, x2, y2 format and apply NMS
        try:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                scores.tolist(), 
                score_threshold=0.0,  # Already filtered by confidence
                nms_threshold=threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                return [detections[i] for i in indices]
            else:
                return detections
        except:
            # Fallback - return all detections
            return detections
    
    def _project_detection_to_erp(self, detection: Dict, tile_params: Dict, intrinsics: np.ndarray,
                                 tile_size: int, erp_h: int, erp_w: int) -> Optional[Dict]:
        """Project tile detection back to ERP coordinates."""
        
        x1, y1, x2, y2 = detection['box']
        
        # Convert tile box corners to ERP coordinates
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        erp_corners = []
        
        for tile_x, tile_y in corners:
            # Convert tile pixel to camera ray
            focal_length = intrinsics[0, 0]
            x_cam = (tile_x - tile_size / 2) / focal_length
            y_cam = (tile_y - tile_size / 2) / focal_length
            z_cam = 1.0
            
            # Transform to world coordinates using tile extrinsics
            # (Simplified - assumes identity extrinsics for now)
            world_ray = np.array([x_cam, y_cam, z_cam])
            world_ray = world_ray / np.linalg.norm(world_ray)
            
            # Convert to ERP coordinates
            erp_u, erp_v = SphericalProjection.xyz_to_erp(
                world_ray[0], world_ray[1], world_ray[2], erp_w, erp_h
            )
            
            erp_corners.append((erp_u, erp_v))
        
        # Find bounding box of ERP corners
        erp_xs = [c[0] for c in erp_corners]
        erp_ys = [c[1] for c in erp_corners]
        
        erp_x1, erp_x2 = min(erp_xs), max(erp_xs)
        erp_y1, erp_y2 = min(erp_ys), max(erp_ys)
        
        # Validate bounds
        if erp_x2 - erp_x1 > erp_w / 2:  # Box spans more than half width - likely seam issue
            return None
        
        erp_x1 = max(0, min(erp_w, erp_x1))
        erp_x2 = max(0, min(erp_w, erp_x2))
        erp_y1 = max(0, min(erp_h, erp_y1))
        erp_y2 = max(0, min(erp_h, erp_y2))
        
        return {
            'box': [erp_x1, erp_y1, erp_x2, erp_y2],
            'confidence': detection['confidence'],
            'label': detection['label'],
            'label_id': detection['label_id'],
            'source_tile': tile_params['tile_id']
        }
    
    def _remove_duplicate_detections(self, detections: List[Dict], threshold: float) -> List[Dict]:
        """Remove duplicate detections from multiple tiles."""
        if len(detections) <= 1:
            return detections
        
        # Group by label
        label_groups = {}
        for det in detections:
            label = det['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(det)
        
        # Apply NMS within each label group
        filtered = []
        for label, group in label_groups.items():
            filtered.extend(self._apply_nms(group, threshold))
        
        # Sort by confidence
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
        
        return filtered
    
    def _filter_with_hint(self, detections: List[Dict], hint_mask: np.ndarray, H: int, W: int) -> List[Dict]:
        """Filter detections using hint mask."""
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            
            # Check if detection center overlaps with hint
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            if 0 <= center_x < W and 0 <= center_y < H:
                if hint_mask[center_y, center_x] > 0.5:  # Hint suggests object here
                    filtered.append(det)
        
        return filtered
    
    def _fix_erp_seam_detections(self, detections: List[Dict], width: int) -> List[Dict]:
        """Fix detection boxes that cross the ERP longitude seam."""
        fixed = []
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            
            # Handle longitude wrapping
            if x1 < 0:
                x1 += width
            if x2 < 0:
                x2 += width
            if x1 >= width:
                x1 -= width  
            if x2 >= width:
                x2 -= width
            
            # Ensure x1 <= x2 (handle wraparound)
            if x1 > x2:
                # Box spans seam - split into two
                fixed.append({
                    **det,
                    'box': [x1, y1, width, y2],
                    'seam_box_part': 'right'
                })
                fixed.append({
                    **det,
                    'box': [0, y1, x2, y2],
                    'seam_box_part': 'left'
                })
            else:
                det['box'] = [x1, y1, x2, y2]
                fixed.append(det)
        
        return fixed
    
    def _create_detection_preview(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Create visualization of detections."""
        
        preview = img.copy()
        H, W = img.shape[:2]
        
        # Draw detection boxes
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            confidence = det['confidence']
            label = det['label']
            
            # Convert to integer coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Choose color based on label hash
            color_idx = hash(label) % 6
            colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
            color = colors[color_idx]
            
            # Draw bounding box
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            text = f"{label}: {confidence:.2f}"
            font_scale = 0.5
            thickness = 1
            
            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw text background
            cv2.rectangle(preview, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            
            # Draw text
            cv2.putText(preview, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 0, 0), thickness)
        
        return preview.astype(np.float32)
    
    def _create_detection_summary(self, detections: List[Dict], queries: List[str]) -> str:
        """Create text summary of detections."""
        
        if not detections:
            return f"No objects detected for queries: {', '.join(queries)}"
        
        # Count by label
        label_counts = {}
        for det in detections:
            label = det['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Create summary
        summary_lines = [
            f"Detected {len(detections)} objects:",
            ""
        ]
        
        for label, count in sorted(label_counts.items()):
            avg_conf = np.mean([d['confidence'] for d in detections if d['label'] == label])
            summary_lines.append(f"• {label}: {count} (avg conf: {avg_conf:.2f})")
        
        return "\n".join(summary_lines)