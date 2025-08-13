"""
Complete all-in-one node for 360° panorama to 3D geometry conversion.
Integrates the entire pipeline from ERP input to final mesh output.
"""

import os
import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

# Import all pipeline components
from .spherical_nodes import Pano_TileSampler_Spherical, MoGe_PerTile_Geometry, Depth_Normal_Stitcher_360
from .detection_nodes import OWLViT_Detect_360
from .matting_nodes import Boxes_To_ZIM_Mattes, Detection_Mask_Combiner
from .layer_nodes import Sky_Background_Splitter, Layer_Builder_360
from .inpainting_nodes import Layer_Complete_360, Layer_Alpha_Refiner
from .depth_alignment_nodes import Depth_Align_Layers
from .mesh_nodes import Layer_Mesher_Spherical
import folder_paths

import comfy.model_management as mm

log = logging.getLogger(__name__)

class Pano360_To_Geometry_Complete:
    """Complete 360° panorama to 3D geometry conversion in a single node."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "erp_image": ("IMAGE", {
                    "tooltip": "Input equirectangular panorama image in 2:1 aspect ratio. This is the only required input - everything else is automatically processed."
                }),
                "moge_model": ("MOGEMODEL", {
                    "tooltip": "Pre-loaded MoGe model for depth estimation. Use 'Load MoGe Model' node to load the model first."
                }),
                "quality_preset": (["fast", "balanced", "high", "ultra"], {
                    "default": "balanced",
                    "tooltip": "Processing quality vs speed: 'fast' for quick previews, 'balanced' for good quality, 'high' for detailed results, 'ultra' for maximum quality."
                }),
                "mesh_resolution": ("INT", {
                    "default": 1024, "min": 512, "max": 2048, "step": 128,
                    "tooltip": "Final mesh resolution. Higher values = more detail but slower processing and larger files. Preset overrides this if not custom."
                }),
                "enable_layering": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable object detection and layer separation for more sophisticated 3D reconstruction. Slower but higher quality results."
                }),
            },
            "optional": {
                "detection_queries": ("STRING", {
                    "default": "mountain, tree, building, rock, person, vehicle",
                    "multiline": True,
                    "tooltip": "Objects to detect for layering (only used if layering enabled). Comma-separated list of objects in natural language."
                }),
                "detection_confidence": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.5, "step": 0.001,
                    "tooltip": "Confidence threshold for object detection. Lower values detect more objects but may include false positives."
                }),
                "depth_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                    "tooltip": "Global depth scaling factor. 1.0 = original scale, >1.0 = exaggerated depth, <1.0 = flattened depth."
                }),
                "material_preset": (["matte", "metallic", "glass", "custom"], {
                    "default": "matte",
                    "tooltip": "Material appearance: 'matte' for natural scenes, 'metallic' for reflective surfaces, 'glass' for transparency effects."
                }),
                "layer_separation_mode": (["auto", "sky_only", "objects_only"], {
                    "default": "auto",
                    "tooltip": "Layering strategy: 'auto' detects objects and splits sky/background, 'sky_only' skips object detection, 'objects_only' emphasizes detected objects."
                }),
                "remove_edge_artifacts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove mesh artifacts at depth discontinuities. Recommended for clean geometry but may create small holes."
                }),
                "smooth_normals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate smooth vertex normals for better lighting. Creates softer appearance but may blur sharp edges."
                }),
                # Grid/tiling overrides
                "grid_yaw_override": ("INT", {
                    "default": 0, "min": 0, "max": 24, "step": 1,
                    "tooltip": "Override horizontal tile count. 0 uses preset value."
                }),
                "grid_pitch_override": ("INT", {
                    "default": 0, "min": 0, "max": 12, "step": 1,
                    "tooltip": "Override vertical tile count. 0 uses preset value."
                }),
                "tile_resolution_override": ("INT", {
                    "default": 0, "min": 0, "max": 2048, "step": 64,
                    "tooltip": "Override tile resolution. 0 uses preset value."
                }),
                "stitcher_height_override": ("INT", {
                    "default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "Override output ERP height. 0 uses preset value."
                }),
                # Optional export
                "export_format": (["none", "glb", "ply"], {
                    "default": "none",
                    "tooltip": "Optionally export the mesh to disk and include the path in the report."
                }),
                "export_prefix": ("STRING", {
                    "default": "3D/MoGe360_Complete",
                    "tooltip": "Filename prefix for exported meshes under ComfyUI's output directory."
                }),
                # Debug/diagnostics (appended to end for backward compatibility)
                "enable_depth_diagnostics": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If enabled, computes depth statistics after spherical stitching and appends them to the progress report."
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("final_mesh", "depth_map", "normal_map", "process_preview", "progress_report")
    OUTPUT_TOOLTIPS = (
        "Complete 3D mesh with textures and materials ready for export or further processing in 3D applications.",
        "Final depth map visualization showing the depth values used for mesh generation with color coding.",
        "Final normal map visualization showing surface normals as RGB colors for lighting verification.",
        "Processing overview showing tiles, detections, layers, and other pipeline stages for debugging.",
        "Detailed text report of the entire processing pipeline including timing, statistics, and any issues encountered."
    )
    FUNCTION = "process_complete_pipeline"
    CATEGORY = "MoGe360/Complete"
    DESCRIPTION = "Complete 360° panorama to 3D geometry pipeline in a single node. Takes an ERP image and outputs a textured 3D mesh with optional object detection and layer separation."

    def process_complete_pipeline(self, erp_image: torch.Tensor, moge_model, quality_preset: str, 
                                mesh_resolution: int, enable_layering: bool,
                                detection_queries: str = "mountain, tree, building, rock, person, vehicle",
                                detection_confidence: float = 0.01, depth_scale: float = 1.0,
                                material_preset: str = "matte", layer_separation_mode: str = "auto",
                                remove_edge_artifacts: bool = True, smooth_normals: bool = True,
                                grid_yaw_override: int = 0, grid_pitch_override: int = 0,
                                tile_resolution_override: int = 0, stitcher_height_override: int = 0,
                                enable_depth_diagnostics: bool = False,
                                export_format: str = "none", export_prefix: str = "3D/MoGe360_Complete"):
        
        device = mm.get_torch_device()
        start_time = time.time()
        
        # Initialize progress tracking
        progress_log = []
        progress_log.append(f"=== 360° PANORAMA TO GEOMETRY PIPELINE ===")
        progress_log.append(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        progress_log.append(f"Quality preset: {quality_preset}")
        progress_log.append(f"Layering enabled: {enable_layering}")
        progress_log.append(f"Input image shape: {erp_image.shape}")
        progress_log.append("")

        # Validate inputs
        valid_ok, validation_msg = self._validate_inputs(erp_image)
        if not valid_ok:
            progress_log.append(f"Warning: {validation_msg}")
        
        # Backward compatibility: handle saved workflows where widget order shifted
        try:
            if export_format not in {"none", "glb", "ply"}:
                # Likely the old value for export_prefix ended up here
                progress_log.append(
                    f"Note: corrected export params (received export_format='{export_format}'); treating it as export_prefix"
                )
                export_prefix = str(export_format)
                export_format = "none"
        except Exception:
            pass

        
        # Configure settings based on quality preset
        settings = self._configure_quality_preset(quality_preset, mesh_resolution)
        # Apply overrides if provided
        if grid_yaw_override > 0:
            settings["tile_grid_yaw"] = grid_yaw_override
        if grid_pitch_override > 0:
            settings["tile_grid_pitch"] = grid_pitch_override
        if tile_resolution_override > 0:
            settings["tile_resolution"] = tile_resolution_override
        if stitcher_height_override > 0:
            settings["stitcher_height"] = stitcher_height_override
        progress_log.append(f"Configured settings: {settings}")
        progress_log.append("")
        
        try:
            # Stage 1: Spherical MoGe Processing
            stage1_start = time.time()
            progress_log.append("Stage 1/3: Spherical MoGe processing")
            progress_log.append("=" * 50)
            
            spherical_results, stage1_log = self._run_spherical_pipeline(
                erp_image, moge_model, settings
            )
            progress_log.extend(stage1_log)
            
            if spherical_results is None:
                raise RuntimeError("Spherical MoGe processing failed")
            
            tiles, tile_params, erp_depth, erp_normals = spherical_results

            # Optional depth diagnostics (append to progress log)
            try:
                if enable_depth_diagnostics and erp_depth is not None:
                    diag_summary = self._compute_depth_diagnostics_summary(erp_depth)
                    progress_log.append("Depth diagnostics:")
                    for line in diag_summary.split('\n'):
                        progress_log.append(f"  {line}")
                    progress_log.append("")
            except Exception as e:
                progress_log.append(f"Depth diagnostics failed: {e}")
            stage1_time = time.time() - stage1_start
            progress_log.append(f"Stage 1 completed in {stage1_time:.1f}s")
            progress_log.append("")
            # Memory cleanup between stages
            try:
                mm.soft_empty_cache()
            except Exception:
                pass
            
            # Stage 2: Optional Layer Processing
            layer_stack = None
            if enable_layering:
                stage2_start = time.time()
                progress_log.append("Stage 2/3: Layer processing & object detection")
                progress_log.append("=" * 50)
                
                layer_results, stage2_log = self._run_layering_pipeline(
                    erp_image, erp_depth, detection_queries, detection_confidence, settings, layer_separation_mode
                )
                progress_log.extend(stage2_log)
                
                if layer_results is not None:
                    layer_stack = layer_results
                    stage2_time = time.time() - stage2_start
                    progress_log.append(f"Stage 2 completed in {stage2_time:.1f}s")
                else:
                    progress_log.append("Stage 2 failed - continuing with simple mesh generation")
                progress_log.append("")
            else:
                progress_log.append("Stage 2/3: Skipped (layering disabled)")
                progress_log.append("")
            
            # Memory cleanup
            try:
                mm.soft_empty_cache()
            except Exception:
                pass

            # Stage 3: Mesh Generation
            stage3_start = time.time()
            progress_log.append("Stage 3/3: Mesh generation")
            progress_log.append("=" * 50)
            
            layer_selection = {
                "auto": "all",
                "sky_only": "sky_only",
                "objects_only": "objects_only",
            }.get(layer_separation_mode, "all")
            mesh_results, stage3_log = self._run_mesh_generation(
                layer_stack, erp_image, erp_depth, erp_normals, 
                settings, material_preset, remove_edge_artifacts, smooth_normals, depth_scale,
                layer_selection, export_format, export_prefix
            )
            progress_log.extend(stage3_log)
            
            if mesh_results is None:
                raise RuntimeError("Mesh generation failed")
            
            final_mesh, mesh_depth_vis, mesh_normals_vis = mesh_results
            stage3_time = time.time() - stage3_start
            progress_log.append(f"Stage 3 completed in {stage3_time:.1f}s")
            progress_log.append("")
            
            # Create process preview
            process_preview = self._create_process_preview(
                erp_image, tiles, erp_depth, erp_normals, layer_stack, device
            )
            
            # Final summary
            total_time = time.time() - start_time
            progress_log.append("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            progress_log.append(f"Total processing time: {total_time:.1f}s")
            progress_log.append(f"Final mesh: {len(final_mesh.vertices) if final_mesh else 0} vertices, {len(final_mesh.faces) if final_mesh else 0} faces")
            progress_log.append(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0:.1f} GB")
            
            log.info(f"Complete pipeline finished in {total_time:.1f}s")
            
            return (
                final_mesh,
                mesh_depth_vis if mesh_depth_vis is not None else erp_depth,
                mesh_normals_vis if mesh_normals_vis is not None else erp_normals,
                process_preview,
                "\n".join(progress_log)
            )
            
        except Exception as e:
            error_time = time.time() - start_time
            progress_log.append(f"=== PIPELINE FAILED ===")
            progress_log.append(f"Error after {error_time:.1f}s: {str(e)}")
            progress_log.append(f"Check ComfyUI console for detailed error information")
            
            log.error(f"Complete pipeline failed: {e}")
            
            # Return empty/fallback results
            import trimesh
            empty_mesh = trimesh.Trimesh()
            empty_img = torch.zeros((1, 512, 1024, 3), device=device, dtype=torch.float32)
            
            return (
                empty_mesh,
                empty_img,
                empty_img, 
                empty_img,
                "\n".join(progress_log)
            )

    def _validate_inputs(self, erp_image: torch.Tensor) -> Tuple[bool, str]:
        """Validate ERP input shape and aspect ratio."""
        try:
            _, H, W, C = erp_image.shape
        except Exception:
            return False, "Invalid image tensor shape; expected [B,H,W,C]"
        if C != 3:
            return False, f"Expected 3 channels, got {C}"
        if H <= 0 or W <= 0:
            return False, "Non-positive image dimensions"
        if abs((W / max(1, H)) - 2.0) > 0.05:
            return False, f"Input is not ~2:1 ERP (W/H={W/H:.2f})"
        return True, "OK"
    
    def _configure_quality_preset(self, preset: str, mesh_resolution: int) -> Dict[str, Any]:
        """Configure processing parameters based on quality preset."""
        
        presets = {
            "fast": {
                "tile_grid_yaw": 3,
                "tile_grid_pitch": 2,
                "tile_fov": 100.0,
                "tile_overlap": 10.0,
                "tile_resolution": 512,
                "moge_resolution_level": 7,
                "moge_batch_size": 6,
                "mesh_resolution": 512,
                "stitcher_height": 512,
                "blend_mode": "weighted",
                "edge_feather": 0.05,
            },
            "balanced": {
                "tile_grid_yaw": 6,
                "tile_grid_pitch": 3,
                "tile_fov": 100.0,
                "tile_overlap": 15.0,
                "tile_resolution": 768,
                "moge_resolution_level": 9,
                "moge_batch_size": 4,
                "mesh_resolution": 1024,
                "stitcher_height": 1024,
                "blend_mode": "weighted",
                "edge_feather": 0.1,
            },
            "high": {
                "tile_grid_yaw": 9,
                "tile_grid_pitch": 4,
                "tile_fov": 90.0,
                "tile_overlap": 20.0,
                "tile_resolution": 768,
                "moge_resolution_level": 10,
                "moge_batch_size": 3,
                "mesh_resolution": 1536,
                "stitcher_height": 1536,
                "blend_mode": "feather",
                "edge_feather": 0.15,
            },
            "ultra": {
                "tile_grid_yaw": 12,
                "tile_grid_pitch": 6,
                "tile_fov": 90.0,
                "tile_overlap": 25.0,
                "tile_resolution": 1024,
                "moge_resolution_level": 11,
                "moge_batch_size": 2,
                "mesh_resolution": 2048,
                "stitcher_height": 2048,
                "blend_mode": "feather",
                "edge_feather": 0.2,
            }
        }
        
        settings = presets.get(preset, presets["balanced"]).copy()
        
        # Override mesh resolution if provided
        if mesh_resolution != 1024:  # Default value
            settings["mesh_resolution"] = mesh_resolution
        
        return settings
    
    def _run_spherical_pipeline(self, erp_image: torch.Tensor, moge_model, settings: Dict) -> Tuple[Optional[Tuple], List[str]]:
        """Execute the spherical MoGe processing pipeline."""
        
        progress_log = []
        
        try:
            # Initialize nodes
            tile_sampler = Pano_TileSampler_Spherical()
            moge_processor = MoGe_PerTile_Geometry()
            depth_stitcher = Depth_Normal_Stitcher_360()
            
            # Step 1: Sample tiles
            progress_log.append(f"Sampling {settings['tile_grid_yaw']}x{settings['tile_grid_pitch']} perspective tiles...")
            
            tiles, tile_params, preview = tile_sampler.sample_tiles(
                erp_image=erp_image,
                grid_yaw=settings["tile_grid_yaw"],
                grid_pitch=settings["tile_grid_pitch"],
                tile_fov=settings["tile_fov"],
                overlap=settings["tile_overlap"],
                tile_resolution=settings["tile_resolution"]
            )
            
            progress_log.append(f"Sampled {tiles.shape[0]} tiles at {settings['tile_resolution']}px resolution")
            
            # Step 2: Process tiles through MoGe
            progress_log.append(f"Processing tiles through MoGe (level {settings['moge_resolution_level']})...")
            
            tile_depths, tile_normals, tile_geometry = moge_processor.process_tiles(
                moge_model=moge_model,
                tiles=tiles,
                tile_params=tile_params,
                resolution_level=settings["moge_resolution_level"],
                batch_size=settings["moge_batch_size"],
                lock_fov=True,
                create_preview=False
            )
            
            progress_log.append(f"Processed {tile_depths.shape[0]} tiles successfully")
            
            # Step 3: Stitch to ERP
            progress_log.append(f"Stitching tiles to {settings['stitcher_height']}px ERP...")
            
            # Extract stitch_map from tile_params
            stitch_map = tile_params.get('stitch_map', {})
            
            erp_depth, erp_normals, stitch_preview = depth_stitcher.stitch_geometry(
                tile_depths=tile_depths,
                tile_normals=tile_normals,
                tile_params=tile_params,
                stitch_map=stitch_map,
                output_height=settings["stitcher_height"],
                blend_mode=settings["blend_mode"],
                edge_feather=settings["edge_feather"],
                debug_seams=False,
                create_preview=True
            )
            
            progress_log.append(f"Stitched to {erp_depth.shape[1]}x{erp_depth.shape[2]} ERP depth map")
            
            return (tiles, tile_params, erp_depth, erp_normals), progress_log
            
        except Exception as e:
            progress_log.append(f"ERROR in spherical pipeline: {str(e)}")
            log.error(f"Spherical pipeline failed: {e}")
            return None, progress_log
    
    def _run_layering_pipeline(self, erp_image: torch.Tensor, erp_depth: torch.Tensor, 
                             detection_queries: str, detection_confidence: float, 
                             settings: Dict, separation_mode: str = "auto") -> Tuple[Optional[Dict], List[str]]:
        """Execute the object detection and layer processing pipeline."""
        
        progress_log = []
        
        try:
            # Initialize nodes
            detector = OWLViT_Detect_360()
            matter = Boxes_To_ZIM_Mattes()
            sky_splitter = Sky_Background_Splitter()
            layer_builder = Layer_Builder_360()
            layer_completer = Layer_Complete_360()
            alpha_refiner = Layer_Alpha_Refiner()
            depth_aligner = Depth_Align_Layers()
            
            # Step 1: Object Detection (optional)
            detection_boxes = {'detection_count': 0}
            if separation_mode != "sky_only":
                progress_log.append(f"Detecting objects: {detection_queries[:50]}...")
                detection_boxes, detection_preview, detection_summary = detector.detect_objects(
                    erp_image=erp_image,
                    text_queries=detection_queries,
                    confidence_threshold=detection_confidence,
                    nms_threshold=0.5,
                    erp_mode="circular_padding",
                    max_detections=20,
                    model_size="base"
                )
                progress_log.append(f"Detected {detection_boxes.get('detection_count', 0)} objects")
            else:
                progress_log.append("Object detection skipped (sky_only mode)")
            
            # Step 2: Generate masks (if objects detected)
            object_masks = None
            if detection_boxes.get('detection_count', 0) > 0:
                progress_log.append("Converting detections to masks...")
                
                # Create object masks from detections
                object_masks = matter.create_masks(
                    erp_image=erp_image,
                    detection_boxes=detection_boxes,
                    matting_method="grabcut",
                    mask_expansion=1.2,
                    edge_feather=2.0,
                    min_mask_size=100
                )
                
                progress_log.append(f"Generated {object_masks[0].shape[0]} object masks")
            
            # Step 3: Sky/Background Split
            progress_log.append("Splitting sky and background...")
            
            sky_mask, background_mask, split_preview = sky_splitter.split_sky_background(
                erp_image=erp_image,
                method="auto",
                sky_threshold=0.6,
                horizon_latitude=0.0,
                feather_amount=0.05,
                min_sky_height=0.3
            )
            
            progress_log.append("Sky/background split completed")
            
            # Step 4: Build Layers
            progress_log.append("Building layer stack...")
            
            layer_stack = layer_builder.build_layers(
                erp_image=erp_image,
                erp_depth=erp_depth,
                sky_mask=sky_mask,
                background_mask=background_mask,
                sky_depth_value=1000.0,
                depth_scale_bg=1.0,
                layer_priority="sky_back",
                edge_cleanup=True,
                object_masks=object_masks[0] if object_masks else None
            )
            
            layer_count = len(layer_stack[0]['layers'])
            progress_log.append(f"Built {layer_count} layers")
            
            # Step 5: Complete Layers (Inpainting)
            progress_log.append("Completing layers with inpainting...")
            
            completed_layers = layer_completer.complete_layers(
                layer_stack=layer_stack[0],
                inpaint_method="opencv_telea",
                inpaint_radius=5,
                blend_strength=0.8,
                seam_protection=True,
                quality_mode="balanced"
            )
            
            progress_log.append("Layer completion finished")
            try:
                total_inpainted = sum(l.get('metadata', {}).get('inpainted_pixels', 0) for l in completed_layers[0]['layers'])
                progress_log.append(f"Inpainted pixels (total): {total_inpainted}")
            except Exception:
                pass
            
            # Step 6: Refine Alpha Masks
            progress_log.append("Refining alpha masks...")
            
            refined_layers = alpha_refiner.refine_alphas(
                layer_stack=completed_layers[0],
                smoothing_method="bilateral",
                smoothing_strength=2.0,
                edge_threshold=0.1,
                cleanup_small_regions=True,
                min_region_size=50
            )
            
            progress_log.append("Alpha refinement completed")
            
            # Step 7: Align Depths
            progress_log.append("Aligning layer depths...")
            
            aligned_layers, alignment_report = depth_aligner.align_layer_depths(
                layer_stack=refined_layers[0],
                reference_depth=erp_depth,
                alignment_method="overlap_based",
                depth_tolerance=0.05,
                min_overlap_pixels=100,
                alignment_strength=0.8,
                preserve_relative_order=True
            )
            
            progress_log.append("Depth alignment completed")
            progress_log.append(f"Alignment summary: {alignment_report.split('Overall consistency score:')[-1].strip() if 'Overall consistency score:' in alignment_report else 'completed'}")
            
            return aligned_layers, progress_log
            
        except Exception as e:
            progress_log.append(f"ERROR in layering pipeline: {str(e)}")
            log.error(f"Layering pipeline failed: {e}")
            return None, progress_log
    
    def _run_mesh_generation(self, layer_stack: Optional[Dict], erp_image: torch.Tensor, 
                           erp_depth: torch.Tensor, erp_normals: torch.Tensor,
                           settings: Dict, material_preset: str, remove_edge_artifacts: bool, 
                           smooth_normals: bool, depth_scale: float,
                           layer_selection: str, export_format: str, export_prefix: str) -> Tuple[Optional[Tuple], List[str]]:
        """Execute the mesh generation pipeline."""
        
        progress_log = []
        
        try:
            # Initialize mesh generator
            mesher = Layer_Mesher_Spherical()
            
            # Configure material properties
            metallic_factor, roughness_factor = self._get_material_properties(material_preset)
            
            if layer_stack is not None:
                # Use layered mesh generation
                progress_log.append(f"Generating mesh from {len(layer_stack['layers'])} layers (selection: {layer_selection})...")
                
                # For proper terrain detail, process layers individually first
                # then merge - this preserves mountain depth variation
                merge_layers_flag = layer_selection == "all"  # Only merge if using all layers
                
                mesh, depth_vis, normals_vis = mesher.create_spherical_mesh(
                    layer_stack=layer_stack,
                    mesh_resolution=settings["mesh_resolution"],
                    layer_selection=layer_selection,
                    depth_scale=depth_scale,
                    remove_edge=remove_edge_artifacts,
                    smooth_normals=smooth_normals,
                    metallic_factor=metallic_factor,
                    roughness_factor=roughness_factor,
                    merge_layers=merge_layers_flag
                )
                
                progress_log.append("Layered mesh generation completed")
                
            else:
                # Use simple mesh generation from depth
                progress_log.append("Generating simple mesh from depth map...")
                
                # Create simple alpha mask from depth with better coverage
                depth_np = erp_depth[0, :, :, 0].cpu().numpy()
                
                # Use adaptive threshold based on depth statistics
                valid_depths = depth_np[depth_np > 0]
                if len(valid_depths) > 0:
                    depth_threshold = max(0.001, valid_depths.min() * 0.1)  # Use 10% of minimum valid depth
                    progress_log.append(f"Using depth threshold: {depth_threshold:.4f} (min valid: {valid_depths.min():.4f})")
                else:
                    depth_threshold = 0.01
                    progress_log.append("No valid depths found, using default threshold: 0.01")
                
                alpha_mask = (depth_np > depth_threshold).astype(np.float32)
                
                # Ensure we have sufficient coverage
                coverage = alpha_mask.mean()
                progress_log.append(f"Alpha mask coverage: {coverage:.2%} of image")
                
                if coverage < 0.1:  # Less than 10% coverage
                    # Fall back to more aggressive threshold
                    alpha_mask = (depth_np > 0.0001).astype(np.float32)
                    coverage = alpha_mask.mean()
                    progress_log.append(f"Expanded alpha mask coverage to: {coverage:.2%}")
                
                alpha_tensor = torch.from_numpy(alpha_mask).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(erp_depth.device)
                
                mesh, depth_vis, normals_vis = mesher.create_spherical_mesh(
                    layer_rgb=erp_image,
                    layer_depth=erp_depth,
                    layer_alpha=alpha_tensor,
                    mesh_resolution=settings["mesh_resolution"],
                    depth_scale=depth_scale,
                    remove_edge=remove_edge_artifacts,
                    smooth_normals=smooth_normals,
                    metallic_factor=metallic_factor,
                    roughness_factor=roughness_factor
                )
                
                progress_log.append("Simple mesh generation completed")
            
            if mesh is not None:
                vertex_count = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
                face_count = len(mesh.faces) if hasattr(mesh, 'faces') else 0
                progress_log.append(f"Generated mesh: {vertex_count} vertices, {face_count} faces")
            else:
                progress_log.append("WARNING: Mesh generation returned None")
            
            # Optional export
            if export_format != "none" and mesh is not None:
                try:
                    out_dir, name, counter, subfolder, prefix = folder_paths.get_save_image_path(
                        export_prefix, folder_paths.get_output_directory()
                    )
                    out_path = Path(out_dir) / f"{name}_{counter:05d}_.{export_format}"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    mesh.export(out_path)
                    progress_log.append(f"Exported mesh: {out_path}")
                except Exception as e:
                    progress_log.append(f"WARNING: Mesh export failed: {e}")

            return (mesh, depth_vis, normals_vis), progress_log
            
        except Exception as e:
            progress_log.append(f"ERROR in mesh generation: {str(e)}")
            log.error(f"Mesh generation failed: {e}")
            return None, progress_log
    
    def _get_material_properties(self, material_preset: str) -> Tuple[float, float]:
        """Get metallic and roughness factors for material preset."""
        
        presets = {
            "matte": (0.0, 0.9),      # Non-metallic, very rough
            "metallic": (1.0, 0.2),   # Fully metallic, smooth
            "glass": (0.0, 0.1),      # Non-metallic, very smooth
            "custom": (0.0, 0.8),     # Default values for custom
        }
        
        return presets.get(material_preset, presets["matte"])
    
    def _create_process_preview(self, erp_image: torch.Tensor, tiles: torch.Tensor, 
                              erp_depth: torch.Tensor, erp_normals: torch.Tensor, 
                              layer_stack: Optional[Dict], device) -> torch.Tensor:
        """Create a preview image showing the processing pipeline stages."""
        
        try:
            import cv2
            
            B, H, W, C = erp_image.shape
            
            # Create a 2x2 grid preview
            preview_h, preview_w = H // 2, W // 2
            preview = np.zeros((preview_h * 2, preview_w * 2, 3), dtype=np.float32)
            
            # Top-left: Original image
            original = cv2.resize(erp_image[0].cpu().numpy(), (preview_w, preview_h))
            preview[:preview_h, :preview_w] = original
            
            # Top-right: Depth map
            if erp_depth is not None:
                depth = erp_depth[0, :, :, 0].cpu().numpy()
                if depth.max() > depth.min():
                    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
                else:
                    depth_norm = depth
                depth_colored = cv2.resize(depth_norm, (preview_w, preview_h))
                preview[:preview_h, preview_w:] = np.stack([depth_colored] * 3, axis=-1)
            
            # Bottom-left: Normal map
            if erp_normals is not None:
                normals = erp_normals[0].cpu().numpy()
                if normals.shape[2] == 3:
                    normals_norm = (normals + 1) / 2  # Convert from [-1,1] to [0,1]
                    normals_resized = cv2.resize(normals_norm, (preview_w, preview_h))
                    preview[preview_h:, :preview_w] = normals_resized
            
            # Bottom-right: Layer info or tiles preview
            if layer_stack is not None:
                # Show layer count as text overlay
                info_img = np.zeros((preview_h, preview_w, 3), dtype=np.float32)
                layer_count = len(layer_stack.get('layers', []))
                # Simple colored blocks representing layers
                block_height = preview_h // max(1, layer_count)
                for i in range(min(layer_count, preview_h // 10)):
                    color = [(i * 0.3) % 1.0, 0.7, 0.5]  # HSV-like colors
                    y_start = i * block_height
                    y_end = min((i + 1) * block_height, preview_h)
                    info_img[y_start:y_end, :preview_w//4] = color
                preview[preview_h:, preview_w:] = info_img
            else:
                # Show first tile as preview
                if tiles is not None and len(tiles) > 0:
                    tile_preview = cv2.resize(tiles[0].cpu().numpy(), (preview_w, preview_h))
                    preview[preview_h:, preview_w:] = tile_preview
            
            return torch.from_numpy(preview).unsqueeze(0).to(device)
            
        except Exception as e:
            log.warning(f"Failed to create process preview: {e}")
            # Return simple gradient as fallback
            fallback = torch.zeros((1, 512, 1024, 3), device=device, dtype=torch.float32)
            return fallback

    def _compute_depth_diagnostics_summary(self, erp_depth: torch.Tensor) -> str:
        """Compute compact depth stats similar to the debug node and return as multiline string."""
        try:
            d = erp_depth[0, :, :, 0].detach().cpu().float().numpy()
        except Exception as e:
            return f"Invalid depth tensor: {getattr(erp_depth, 'shape', None)} ({e})"

        valid = d > 0
        valid_ratio = float(valid.mean()) if valid.size > 0 else 0.0

        if valid.any():
            vals = d[valid]
            dmin = float(vals.min())
            dmax = float(vals.max())
            dmed = float(np.median(vals))
            p = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99]).astype(float)
            pct_labels = ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]
            pct_str = ", ".join([f"{n}:{v:.6g}" for n, v in zip(pct_labels, p)])
        else:
            dmin = dmax = dmed = 0.0
            pct_str = "p01:0, p05:0, p25:0, p50:0, p75:0, p95:0, p99:0"

        lines = [
            f"valid_ratio: {valid_ratio:.3%}",
            f"min/median/max: {dmin:.6g} / {dmed:.6g} / {dmax:.6g}",
            f"percentiles: {pct_str}",
        ]
        return "\n".join(lines)
