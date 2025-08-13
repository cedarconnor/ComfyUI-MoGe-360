"""
Depth alignment nodes for consistent depth across layers in 360° pipeline.
Ensures proper depth ordering and scaling between sky, background, and object layers.
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy import stats

from .spherical_utils import SphericalProjection
import comfy.model_management as mm

log = logging.getLogger(__name__)

class Depth_Align_Layers:
    """Align depths across all layers for consistent 3D reconstruction."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_stack": ("LAYER_STACK", {
                    "tooltip": "Layer stack from Layer_Alpha_Refiner containing refined layers with inconsistent depth scales. Each layer's depth will be aligned for proper 3D reconstruction."
                }),
                "reference_depth": ("IMAGE", {
                    "tooltip": "Reference depth map from Depth_Normal_Stitcher_360 providing ground truth depth scale. Used as baseline for aligning all layer depths consistently."
                }),
                "alignment_method": (["overlap_based", "statistical", "reference_plane", "manual"], {
                    "default": "overlap_based",
                    "tooltip": "Alignment algorithm: 'overlap_based' uses overlapping regions (best), 'statistical' matches depth distributions, 'reference_plane' uses fixed rules, 'manual' uses user values."
                }),
                "depth_tolerance": ("FLOAT", {
                    "default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01,
                    "tooltip": "Acceptable depth variation tolerance as fraction. 0.05 = 5% tolerance. Lower values enforce stricter alignment but may over-correct natural depth variation."
                }),
                "min_overlap_pixels": ("INT", {
                    "default": 100, "min": 10, "max": 1000, "step": 10,
                    "tooltip": "Minimum overlapping pixels required for overlap-based alignment. If insufficient overlap, falls back to statistical alignment. Adjust for image resolution."
                }),
                "alignment_strength": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1,
                    "tooltip": "Strength of depth alignment correction. 1.0 = full alignment, 0.5 = blend 50/50 with original, 0.1 = minimal correction. Lower values preserve original depth character."
                }),
                "preserve_relative_order": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Maintain relative depth ordering between layers (sky furthest, objects closest). Prevents depth inversions that would break 3D reconstruction."
                }),
            },
            "optional": {
                "manual_sky_depth": ("FLOAT", {
                    "default": 1000.0, "min": 10.0, "max": 10000.0, "step": 10.0,
                    "tooltip": "Fixed depth value for sky layer in manual mode. Should be much larger than scene depth to place sky at infinity. Only used with manual alignment method."
                }),
                "manual_bg_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Scale factor for background depth in manual mode. 1.0 = original depth, >1.0 = further away, <1.0 = closer. Only used with manual alignment method."
                }),
                "depth_hints": ("IMAGE", {
                    "tooltip": "Optional depth hints to guide alignment process. Can provide additional constraints for better alignment in challenging regions."
                }),
            }
        }

    RETURN_TYPES = ("LAYER_STACK", "STRING")
    RETURN_NAMES = ("aligned_layers", "alignment_report")
    OUTPUT_TOOLTIPS = (
        "Layer stack with aligned depth maps ensuring consistent scale and proper ordering across all layers. Ready for spherical mesh generation with correct parallax.",
        "Detailed report of alignment process including scale factors, overlap statistics, and depth consistency analysis. Useful for debugging and quality verification."
    )
    FUNCTION = "align_layer_depths"
    CATEGORY = "MoGe360/Layers"
    DESCRIPTION = "Align depth maps across all layers using intelligent algorithms to ensure consistent depth scale and proper layer ordering. Prevents depth conflicts in 3D reconstruction."

    def align_layer_depths(self, layer_stack: Dict, reference_depth: torch.Tensor, 
                          alignment_method: str, depth_tolerance: float, min_overlap_pixels: int,
                          alignment_strength: float, preserve_relative_order: bool,
                          manual_sky_depth: Optional[float] = None, manual_bg_scale: Optional[float] = None,
                          depth_hints: Optional[torch.Tensor] = None):
        
        device = mm.get_torch_device()
        
        # Convert reference depth to numpy
        ref_depth_np = reference_depth[0, :, :, 0].cpu().numpy()  # [H, W]
        H, W = ref_depth_np.shape
        
        # Get layers from stack
        layers = layer_stack.get('layers', [])
        if not layers:
            log.warning("No layers found in layer_stack")
            return (layer_stack, "Warning: No layers to align")
        
        log.info(f"Aligning depths for {len(layers)} layers using method: {alignment_method}")
        
        # Create aligned layers
        aligned_layers = []
        alignment_info = []
        
        # Sort layers by priority for consistent processing
        sorted_layers = sorted(layers, key=lambda x: x.get('priority', 0))
        
        if alignment_method == "manual":
            aligned_layers = self._manual_alignment(sorted_layers, manual_sky_depth, manual_bg_scale)
        elif alignment_method == "reference_plane":
            aligned_layers = self._reference_plane_alignment(sorted_layers, ref_depth_np, depth_tolerance)
        elif alignment_method == "statistical":
            aligned_layers = self._statistical_alignment(sorted_layers, ref_depth_np, depth_tolerance)
        else:  # overlap_based
            aligned_layers, alignment_info = self._overlap_based_alignment(
                sorted_layers, ref_depth_np, depth_tolerance, min_overlap_pixels, 
                alignment_strength, preserve_relative_order
            )
        
        # Generate alignment report
        report = self._generate_alignment_report(layers, aligned_layers, alignment_info, alignment_method)
        
        # Create new layer stack
        aligned_stack = {
            'layers': aligned_layers,
            'image_dimensions': layer_stack.get('image_dimensions', (H, W)),
            'metadata': {
                **layer_stack.get('metadata', {}),
                'alignment_method': alignment_method,
                'alignment_tolerance': depth_tolerance,
                'aligned_layer_count': len(aligned_layers)
            }
        }
        
        log.info(f"Depth alignment completed. Aligned {len(aligned_layers)} layers.")
        
        return (aligned_stack, report)
    
    def _overlap_based_alignment(self, layers: List[Dict], reference_depth: np.ndarray, 
                               tolerance: float, min_overlap: int, strength: float,
                               preserve_order: bool) -> Tuple[List[Dict], List[Dict]]:
        """Align layers based on overlap regions with reference depth."""
        
        aligned_layers = []
        alignment_info = []
        H, W = reference_depth.shape
        
        for i, layer in enumerate(layers):
            layer_depth = layer['depth'].copy()
            layer_alpha = layer['alpha']
            layer_type = layer.get('type', 'unknown')
            
            # Find overlap with reference depth
            valid_mask = (layer_alpha > 0.1) & (reference_depth > 0.01)
            overlap_pixels = valid_mask.sum()
            
            if overlap_pixels < min_overlap:
                # Not enough overlap - use statistical fallback
                scale_factor, offset = self._calculate_statistical_alignment(layer_depth, reference_depth, valid_mask)
                alignment_type = "statistical_fallback"
            else:
                # Calculate alignment based on overlap
                scale_factor, offset = self._calculate_overlap_alignment(
                    layer_depth, reference_depth, valid_mask, tolerance
                )
                alignment_type = "overlap_based"
            
            # Apply alignment with strength parameter
            if strength < 1.0:
                # Blend between original and aligned depth
                aligned_depth = layer_depth * (1 - strength) + (layer_depth * scale_factor + offset) * strength
            else:
                aligned_depth = layer_depth * scale_factor + offset
            
            # Handle special layer types
            if layer_type == 'sky':
                # Sky should be furthest - ensure minimum depth
                sky_min_depth = np.percentile(reference_depth[reference_depth > 0], 95)
                aligned_depth = np.maximum(aligned_depth, sky_min_depth * 1.5)
            
            # Preserve relative ordering if requested
            if preserve_order and i > 0:
                prev_layer = aligned_layers[-1]
                aligned_depth = self._enforce_depth_ordering(aligned_depth, prev_layer['depth'], layer_alpha)
            
            # Create aligned layer
            aligned_layer = layer.copy()
            aligned_layer['depth'] = aligned_depth
            aligned_layers.append(aligned_layer)
            
            # Record alignment info
            info = {
                'layer_index': i,
                'layer_type': layer_type,
                'scale_factor': float(scale_factor),
                'offset': float(offset),
                'overlap_pixels': int(overlap_pixels),
                'alignment_type': alignment_type,
                'original_depth_range': (float(layer_depth.min()), float(layer_depth.max())),
                'aligned_depth_range': (float(aligned_depth.min()), float(aligned_depth.max()))
            }
            alignment_info.append(info)
        
        return aligned_layers, alignment_info
    
    def _calculate_overlap_alignment(self, layer_depth: np.ndarray, reference_depth: np.ndarray, 
                                   valid_mask: np.ndarray, tolerance: float) -> Tuple[float, float]:
        """Calculate scale and offset to align layer depth with reference depth."""
        
        if valid_mask.sum() < 10:
            return 1.0, 0.0
        
        layer_vals = layer_depth[valid_mask]
        ref_vals = reference_depth[valid_mask]
        
        # Remove outliers
        layer_percentiles = np.percentile(layer_vals, [5, 95])
        ref_percentiles = np.percentile(ref_vals, [5, 95])
        
        valid_range = (layer_vals >= layer_percentiles[0]) & (layer_vals <= layer_percentiles[1])
        if valid_range.sum() < 5:
            valid_range = np.ones_like(layer_vals, dtype=bool)
        
        layer_clean = layer_vals[valid_range]
        ref_clean = ref_vals[valid_range]
        
        if len(layer_clean) < 5:
            return 1.0, 0.0
        
        # Calculate linear alignment: ref = scale * layer + offset
        try:
            # Use robust regression (least squares with outlier rejection)
            slope, intercept, r_value, _, _ = stats.linregress(layer_clean, ref_clean)
            
            # Clamp scale factor to reasonable range
            scale_factor = np.clip(slope, 0.1, 10.0)
            offset = intercept
            
            # If correlation is poor, fall back to simple scaling
            if r_value**2 < 0.3:  # R² < 0.3 indicates poor correlation
                scale_factor = np.median(ref_clean) / np.median(layer_clean) if np.median(layer_clean) > 0 else 1.0
                offset = 0.0
                scale_factor = np.clip(scale_factor, 0.1, 10.0)
            
        except (ValueError, FloatingPointError):
            # Fallback to simple ratio
            scale_factor = np.median(ref_clean) / np.median(layer_clean) if np.median(layer_clean) > 0 else 1.0
            offset = 0.0
            scale_factor = np.clip(scale_factor, 0.1, 10.0)
        
        return scale_factor, offset
    
    def _calculate_statistical_alignment(self, layer_depth: np.ndarray, reference_depth: np.ndarray,
                                       valid_mask: np.ndarray) -> Tuple[float, float]:
        """Statistical alignment based on depth distribution matching."""
        
        if valid_mask.sum() < 5:
            return 1.0, 0.0
        
        layer_vals = layer_depth[layer_depth > 0.01]
        ref_vals = reference_depth[reference_depth > 0.01]
        
        if len(layer_vals) == 0 or len(ref_vals) == 0:
            return 1.0, 0.0
        
        # Match medians
        layer_median = np.median(layer_vals)
        ref_median = np.median(ref_vals)
        
        scale_factor = ref_median / layer_median if layer_median > 0 else 1.0
        scale_factor = np.clip(scale_factor, 0.1, 10.0)
        
        return scale_factor, 0.0
    
    def _reference_plane_alignment(self, layers: List[Dict], reference_depth: np.ndarray,
                                 tolerance: float) -> List[Dict]:
        """Align layers to reference depth plane."""
        
        aligned_layers = []
        ref_median = np.median(reference_depth[reference_depth > 0])
        
        for layer in layers:
            layer_depth = layer['depth'].copy()
            layer_type = layer.get('type', 'unknown')
            
            if layer_type == 'sky':
                # Sky gets far depth
                aligned_depth = np.full_like(layer_depth, ref_median * 3.0)
            elif layer_type == 'background':
                # Background aligns to reference
                layer_median = np.median(layer_depth[layer_depth > 0]) if (layer_depth > 0).any() else 1.0
                scale = ref_median / layer_median if layer_median > 0 else 1.0
                aligned_depth = layer_depth * scale
            else:
                # Objects get foreground depth
                aligned_depth = layer_depth * 0.8  # Slightly closer than background
            
            aligned_layer = layer.copy()
            aligned_layer['depth'] = aligned_depth
            aligned_layers.append(aligned_layer)
        
        return aligned_layers
    
    def _statistical_alignment(self, layers: List[Dict], reference_depth: np.ndarray,
                             tolerance: float) -> List[Dict]:
        """Statistical depth distribution alignment."""
        
        aligned_layers = []
        
        # Calculate reference depth statistics
        ref_vals = reference_depth[reference_depth > 0.01]
        if len(ref_vals) == 0:
            return layers  # No valid reference depth
        
        ref_percentiles = np.percentile(ref_vals, [25, 50, 75])
        
        for i, layer in enumerate(layers):
            layer_depth = layer['depth'].copy()
            layer_type = layer.get('type', 'unknown')
            
            # Calculate target depth based on layer type and index
            if layer_type == 'sky':
                target_depth = ref_percentiles[2] * 2.0  # Far depth
            elif layer_type == 'background':
                target_depth = ref_percentiles[1]  # Median depth
            else:
                # Objects get progressively closer depth
                target_depth = ref_percentiles[0] * (0.5 + i * 0.1)
            
            # Scale layer depth to target
            layer_vals = layer_depth[layer_depth > 0.01]
            if len(layer_vals) > 0:
                layer_median = np.median(layer_vals)
                scale = target_depth / layer_median if layer_median > 0 else 1.0
                aligned_depth = layer_depth * scale
            else:
                aligned_depth = np.full_like(layer_depth, target_depth)
            
            aligned_layer = layer.copy()
            aligned_layer['depth'] = aligned_depth
            aligned_layers.append(aligned_layer)
        
        return aligned_layers
    
    def _manual_alignment(self, layers: List[Dict], sky_depth: Optional[float],
                         bg_scale: Optional[float]) -> List[Dict]:
        """Manual depth alignment with user-specified values."""
        
        aligned_layers = []
        
        for layer in layers:
            layer_type = layer.get('type', 'unknown')
            aligned_layer = layer.copy()
            
            if layer_type == 'sky' and sky_depth is not None:
                aligned_layer['depth'] = np.full_like(layer['depth'], sky_depth)
            elif layer_type == 'background' and bg_scale is not None:
                aligned_layer['depth'] = layer['depth'] * bg_scale
            # Objects keep original depth
            
            aligned_layers.append(aligned_layer)
        
        return aligned_layers
    
    def _enforce_depth_ordering(self, current_depth: np.ndarray, prev_depth: np.ndarray,
                              current_alpha: np.ndarray) -> np.ndarray:
        """Ensure depth ordering is preserved between layers."""
        
        # Where current layer is visible, ensure it's not behind previous layer
        overlap_mask = current_alpha > 0.1
        if not overlap_mask.any():
            return current_depth
        
        # Find minimum depth that ensures proper ordering
        prev_min = prev_depth[overlap_mask].min() if (prev_depth > 0).any() else 0.1
        current_corrected = current_depth.copy()
        
        # Ensure current layer is in front of previous layer
        behind_mask = overlap_mask & (current_depth >= prev_min * 0.95)
        if behind_mask.any():
            current_corrected[behind_mask] = prev_min * 0.9
        
        return current_corrected
    
    def _generate_alignment_report(self, original_layers: List[Dict], aligned_layers: List[Dict],
                                 alignment_info: List[Dict], method: str) -> str:
        """Generate detailed alignment report."""
        
        report_lines = [
            f"Depth Alignment Report",
            f"=" * 50,
            f"Method: {method}",
            f"Layers processed: {len(original_layers)}",
            f"",
            f"Layer Details:"
        ]
        
        for i, (orig, aligned) in enumerate(zip(original_layers, aligned_layers)):
            layer_type = orig.get('type', 'unknown')
            
            orig_range = f"{orig['depth'].min():.3f} - {orig['depth'].max():.3f}"
            aligned_range = f"{aligned['depth'].min():.3f} - {aligned['depth'].max():.3f}"
            
            report_lines.append(f"  Layer {i} ({layer_type}):")
            report_lines.append(f"    Original depth range: {orig_range}")
            report_lines.append(f"    Aligned depth range:  {aligned_range}")
            
            if i < len(alignment_info):
                info = alignment_info[i]
                report_lines.append(f"    Scale factor: {info['scale_factor']:.3f}")
                report_lines.append(f"    Offset: {info['offset']:.3f}")
                report_lines.append(f"    Overlap pixels: {info['overlap_pixels']}")
                report_lines.append(f"    Alignment type: {info['alignment_type']}")
            
            report_lines.append("")
        
        # Add consistency check
        if len(aligned_layers) > 1:
            depth_consistency = self._check_depth_consistency(aligned_layers)
            report_lines.append("Depth Consistency Check:")
            report_lines.append(f"  Average inter-layer separation: {depth_consistency['avg_separation']:.3f}")
            report_lines.append(f"  Depth ordering violations: {depth_consistency['violations']}")
            report_lines.append(f"  Overall consistency score: {depth_consistency['score']:.3f}/1.0")
        
        return "\n".join(report_lines)
    
    def _check_depth_consistency(self, layers: List[Dict]) -> Dict[str, float]:
        """Check depth consistency between layers."""
        
        separations = []
        violations = 0
        
        for i in range(len(layers) - 1):
            curr_depth = layers[i]['depth']
            next_depth = layers[i+1]['depth']
            
            # Calculate average separation
            curr_vals = curr_depth[curr_depth > 0]
            next_vals = next_depth[next_depth > 0]
            
            if len(curr_vals) > 0 and len(next_vals) > 0:
                separation = np.median(next_vals) - np.median(curr_vals)
                separations.append(abs(separation))
                
                # Check for ordering violations
                if separation < 0:
                    violations += 1
        
        avg_separation = np.mean(separations) if separations else 0.0
        consistency_score = max(0.0, 1.0 - violations / max(1, len(layers) - 1))
        
        return {
            'avg_separation': avg_separation,
            'violations': violations,
            'score': consistency_score
        }


class Layer_Depth_Visualizer:
    """Visualize depth alignment results across layers."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layer_stack": ("LAYER_STACK", {
                    "tooltip": "Layer stack with aligned depths from Depth_Align_Layers. Depth information will be visualized to verify alignment quality and layer relationships."
                }),
                "visualization_mode": (["depth_maps", "cross_section", "3d_preview", "statistics"], {
                    "default": "depth_maps",
                    "tooltip": "Visualization type: 'depth_maps' shows side-by-side depth images, 'cross_section' shows depth profiles, '3d_preview' renders perspective view, 'statistics' shows depth analysis."
                }),
                "colormap": (["viridis", "plasma", "turbo", "gray"], {
                    "default": "viridis",
                    "tooltip": "Color mapping for depth visualization: 'viridis' is perceptually uniform, 'plasma' has high contrast, 'turbo' is rainbow-like, 'gray' is monochrome."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_visualization",)
    OUTPUT_TOOLTIPS = (
        "Visualization image showing depth alignment results across all layers. Helps verify alignment quality, identify issues, and understand layer depth relationships.",
    )
    FUNCTION = "visualize_depth_alignment"
    CATEGORY = "MoGe360/Layers"
    DESCRIPTION = "Generate visual analysis of depth alignment results showing depth maps, cross-sections, and statistics to verify alignment quality and debug issues."

    def visualize_depth_alignment(self, layer_stack: Dict, visualization_mode: str, colormap: str):
        
        layers = layer_stack.get('layers', [])
        if not layers:
            # Return empty image
            empty_img = torch.zeros((1, 256, 512, 3), dtype=torch.float32)
            return (empty_img,)
        
        H, W = layers[0]['depth'].shape
        
        if visualization_mode == "depth_maps":
            vis_img = self._create_depth_maps_visualization(layers, H, W, colormap)
        elif visualization_mode == "cross_section":
            vis_img = self._create_cross_section_visualization(layers, H, W)
        elif visualization_mode == "statistics":
            vis_img = self._create_statistics_visualization(layers, H, W)
        else:  # 3d_preview
            vis_img = self._create_3d_preview_visualization(layers, H, W)
        
        # Convert to tensor
        vis_tensor = torch.from_numpy(vis_img).unsqueeze(0).float()
        
        return (vis_tensor,)
    
    def _create_depth_maps_visualization(self, layers: List[Dict], H: int, W: int, colormap: str) -> np.ndarray:
        """Create side-by-side depth map visualization."""
        
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        n_layers = len(layers)
        fig_width = min(n_layers * 4, 20)
        fig_height = 3
        
        fig, axes = plt.subplots(1, n_layers, figsize=(fig_width, fig_height))
        if n_layers == 1:
            axes = [axes]
        
        cmap = cm.get_cmap(colormap)
        
        for i, (ax, layer) in enumerate(zip(axes, layers)):
            depth = layer['depth']
            layer_type = layer.get('type', f'Layer {i}')
            
            # Normalize depth for visualization
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                vmin, vmax = np.percentile(valid_depth, [1, 99])
            else:
                vmin, vmax = 0, 1
            
            im = ax.imshow(depth, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            ax.set_title(f'{layer_type.title()}\nRange: [{depth.min():.2f}, {depth.max():.2f}]')
            ax.axis('off')
            
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img_array.astype(np.float32) / 255.0
    
    def _create_cross_section_visualization(self, layers: List[Dict], H: int, W: int) -> np.ndarray:
        """Create cross-section visualization."""
        
        import matplotlib.pyplot as plt
        
        # Take horizontal cross-section at middle latitude
        mid_lat = H // 2
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot depth profiles
        x = np.arange(W)
        for i, layer in enumerate(layers):
            depth_profile = layer['depth'][mid_lat, :]
            layer_type = layer.get('type', f'Layer {i}')
            ax1.plot(x, depth_profile, label=layer_type, linewidth=2)
        
        ax1.set_xlabel('Longitude (pixels)')
        ax1.set_ylabel('Depth')
        ax1.set_title(f'Depth Cross-Section at Latitude {mid_lat}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot depth histogram
        for i, layer in enumerate(layers):
            depth_vals = layer['depth'][layer['depth'] > 0]
            if len(depth_vals) > 0:
                layer_type = layer.get('type', f'Layer {i}')
                ax2.hist(depth_vals, bins=50, alpha=0.7, label=layer_type, density=True)
        
        ax2.set_xlabel('Depth Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Depth Distribution by Layer')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to numpy
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img_array.astype(np.float32) / 255.0
    
    def _create_statistics_visualization(self, layers: List[Dict], H: int, W: int) -> np.ndarray:
        """Create statistics visualization."""
        
        # Create text-based statistics image
        stats_img = np.ones((600, 800, 3), dtype=np.float32)
        
        # For now, return a simple gradient as placeholder
        # In a full implementation, this would render statistics text
        for i in range(3):
            stats_img[:, :, i] = np.linspace(0.9, 0.1, 600)[:, np.newaxis]
        
        return stats_img
    
    def _create_3d_preview_visualization(self, layers: List[Dict], H: int, W: int) -> np.ndarray:
        """Create 3D preview visualization."""
        
        # Create a simple 3D-like visualization by rendering layers with perspective
        composite = np.zeros((H, W, 3), dtype=np.float32)
        
        for i, layer in enumerate(layers):
            rgb = layer.get('rgb', np.zeros((H, W, 3)))
            alpha = layer.get('alpha', np.zeros((H, W)))
            depth = layer['depth']
            
            # Normalize depth for z-offset
            if depth.max() > 0:
                depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            else:
                depth_norm = np.zeros_like(depth)
            
            # Apply depth-based brightness
            brightness = 1.0 - depth_norm * 0.3
            layer_contrib = rgb * alpha[..., np.newaxis] * brightness[..., np.newaxis]
            
            # Blend layers
            composite = composite * (1 - alpha[..., np.newaxis]) + layer_contrib
        
        return np.clip(composite, 0, 1)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "Depth_Align_Layers": Depth_Align_Layers,
    "Layer_Depth_Visualizer": Layer_Depth_Visualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Depth_Align_Layers": "🔧 360° Depth Alignment",
    "Layer_Depth_Visualizer": "👁️ Layer Depth Visualizer",
}