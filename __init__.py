"""
ComfyUI-MoGe-360: Simplified version with only essential nodes
360° Panorama to 3D Geometry Conversion
"""

import sys
import os
import logging
from pathlib import Path

# Configure only this module's logger instead of global logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Apply performance optimizations
try:
    import torch
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn autotuner
        torch.backends.cudnn.benchmark = True
        # Set optimal number of threads
        torch.set_num_threads(min(os.cpu_count(), 8))
        logger.info("Applied CUDA optimizations")
except Exception as e:
    logger.warning(f"Failed to apply CUDA optimizations: {e}")

# Import simplified nodes
try:
    from .simple_nodes import DownloadAndLoadMoGeModel, Pano360ToGeometrySimple
    logger.info("Successfully imported simplified nodes")
    
except ImportError as e:
    logger.error(f"Failed to import simplified nodes: {e}")
    
    # Create placeholder classes to prevent total failure
    logger.warning("Creating placeholder node classes")
    
    class PlaceholderNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
        
        RETURN_TYPES = ("STRING",)
        FUNCTION = "placeholder"
        CATEGORY = "MoGe360"
        
        def placeholder(self):
            return ("Node not available - check dependencies",)
    
    DownloadAndLoadMoGeModel = PlaceholderNode
    Pano360ToGeometrySimple = PlaceholderNode

# Define node mappings (simplified)
NODE_CLASS_MAPPINGS = {
    "Load MoGe Model": DownloadAndLoadMoGeModel,
    "360° Panorama → Simple Geometry": Pano360ToGeometrySimple,
}

# Define display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "Load MoGe Model": "Load MoGe Model",
    "360° Panorama → Simple Geometry": "360° Panorama → Simple Geometry",
}

# Web directory for JavaScript files (none needed for simplified version)
WEB_DIRECTORY = None

# Validate node registration
def validate_nodes():
    """Validate that all nodes are properly configured"""
    errors = []
    
    for node_name, node_class in NODE_CLASS_MAPPINGS.items():
        if not hasattr(node_class, 'INPUT_TYPES'):
            errors.append(f"Node '{node_name}' missing INPUT_TYPES")
        if not hasattr(node_class, 'FUNCTION'):
            errors.append(f"Node '{node_name}' missing FUNCTION attribute")
        if not hasattr(node_class, 'RETURN_TYPES'):
            errors.append(f"Node '{node_name}' missing RETURN_TYPES attribute")
        if not hasattr(node_class, 'CATEGORY'):
            node_class.CATEGORY = "MoGe360"
    
    if errors:
        for error in errors:
            logger.error(error)
        raise RuntimeError(f"Node validation failed with {len(errors)} errors")
    
    logger.info(f"✓ Successfully validated {len(NODE_CLASS_MAPPINGS)} nodes")
    return True

# Run validation
try:
    validate_nodes()
except RuntimeError as e:
    logger.error(f"Node validation failed: {e}")

# Version information
__version__ = "1.0.1"
__author__ = "Cedar Connor"

# Export required symbols
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY",
    "__version__"
]

logger.info(f"ComfyUI-MoGe-360 v{__version__} loaded successfully")
logger.info(f"Registered {len(NODE_CLASS_MAPPINGS)} essential nodes")