"""
Foreground Object Extraction Tool - Core Modules
"""
from .threshold import (
    convert_to_grayscale,
    apply_gaussian_blur,
    compute_threshold_mask,
    threshold_pipeline,
)
from .grabcut_seg import (
    create_grabcut_mask_from_threshold,
    create_grabcut_mask_from_rect,
    run_grabcut,
    grabcut_to_binary_mask,
    grabcut_pipeline,
)
from .extract import (
    apply_mask_to_image,
    create_transparent_png,
    create_binary_mask_image,
    create_transparent_from_alpha,
    crop_to_foreground,
    composite_on_background,
    extract_foreground,
    save_outputs,
)
from .ai_segment import (
    is_available as ai_is_available,
    ai_alpha_matte,
    alpha_to_binary_mask,
    ai_pipeline,
)
from .utils import (
    load_image,
    validate_image,
    resize_if_needed,
    scale_mask_to_original,
    create_comparison_grid,
    image_to_bytes,
    bytes_to_image,
)

__all__ = [
    # threshold
    "convert_to_grayscale",
    "apply_gaussian_blur",
    "compute_threshold_mask",
    "threshold_pipeline",
    # grabcut_seg
    "create_grabcut_mask_from_threshold",
    "create_grabcut_mask_from_rect",
    "run_grabcut",
    "grabcut_to_binary_mask",
    "grabcut_pipeline",
    # extract
    "apply_mask_to_image",
    "create_transparent_png",
    "create_binary_mask_image",
    "create_transparent_from_alpha",
    "crop_to_foreground",
    "composite_on_background",
    "extract_foreground",
    "save_outputs",
    # ai_segment
    "ai_is_available",
    "ai_alpha_matte",
    "alpha_to_binary_mask",
    "ai_pipeline",
    # utils
    "load_image",
    "validate_image",
    "resize_if_needed",
    "scale_mask_to_original",
    "create_comparison_grid",
    "image_to_bytes",
    "bytes_to_image",
]