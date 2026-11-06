"""
Extraction module: apply binary mask to produce transparent PNG, black background, and mask visualization.
Includes edge feathering for smoother cutout transitions.
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import os


def feather_edges(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to mask edges for smooth alpha transition (feathering).

    This creates a soft transition at the mask boundary, reducing hard edges
    in the final cutout. The blur is applied only near edges using a distance
    transform approach for more natural feathering.

    Args:
        mask: Binary mask, shape (H, W), uint8 {0, 255}
        radius: Feather radius in pixels (odd, >= 1). Use 0 or 1 to disable.

    Returns:
        Feathered mask with smooth transitions, shape (H, W), uint8 {0-255}
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Handle edge cases
    if np.all(mask_binary == 0) or np.all(mask_binary == 255):
        return mask_binary

    if radius <= 1:
        return mask_binary

    # Ensure radius is odd
    if radius % 2 == 0:
        radius += 1

    # Use distance transform for edge-aware feathering
    # Compute distance from foreground to background and vice versa
    dist_fg = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    dist_bg = cv2.distanceTransform(255 - mask_binary, cv2.DIST_L2, 5)

    # Normalize distances to [0, 1] within the feather radius
    # Pixels within 'radius' of the edge get smooth transition
    alpha = np.clip((dist_fg - dist_bg + radius) / (2 * radius), 0, 1)

    # Apply gamma correction for more natural falloff (optional)
    # alpha = alpha ** 0.5  # Uncomment for softer falloff

    # Convert to 0-255
    feathered = (alpha * 255).astype(np.uint8)

    return feathered


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Apply binary mask to image, replacing background with solid color.

    Args:
        image: Input BGR image, shape (H, W, 3), uint8
        mask: Binary mask, shape (H, W), uint8 {0, 255}
        background_color: RGB color for background pixels

    Returns:
        Image with masked background, shape (H, W, 3), uint8
    """
    # Ensure mask is binary 0/255
    mask_binary = (mask > 0).astype(np.uint8)

    # Create background image
    bg = np.full_like(image, background_color, dtype=np.uint8)

    # Blend: foreground from image, background from bg
    result = image * mask_binary[:, :, np.newaxis] + bg * (1 - mask_binary[:, :, np.newaxis])
    return result.astype(np.uint8)


def create_transparent_png(
    image: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 3
) -> np.ndarray:
    """
    Create RGBA image with transparent background from mask.

    Args:
        image: Input BGR image, shape (H, W, 3), uint8
        mask: Binary mask, shape (H, W), uint8 {0, 255}
        feather_radius: Feather radius for smooth alpha transitions (0 or 1 to disable)

    Returns:
        RGBA image, shape (H, W, 4), uint8 (alpha=0 for background, 255 for foreground)
    """
    # Apply feathering for smooth edges
    alpha = feather_edges(mask, radius=feather_radius)

    # Convert BGR to RGBA
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = alpha

    return rgba


def create_transparent_from_alpha(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Create an RGBA image (full frame) using a soft alpha matte directly.

    Args:
        image: Input BGR image, shape (H, W, 3), uint8
        alpha: Soft alpha matte, shape (H, W), uint8 0-255

    Returns:
        RGBA image, shape (H, W, 4), uint8
    """
    if alpha.shape[:2] != image.shape[:2]:
        raise ValueError("alpha and image must have the same height/width")
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = alpha.astype(np.uint8)
    return rgba


def create_binary_mask_image(mask: np.ndarray) -> np.ndarray:
    """
    Create visualization of binary mask (white foreground on black background).

    Args:
        mask: Binary mask, shape (H, W), uint8 {0, 255}

    Returns:
        BGR visualization image, shape (H, W, 3), uint8
    """
    # Convert single channel to 3-channel
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return vis


def extract_foreground_only(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Extract ONLY the foreground object, cropped to its bounding box.
    
    This returns just the foreground pixels with no background (not transparent, not black).
    The result is cropped tightly around the foreground object.

    Args:
        image: Original BGR image, shape (H, W, 3), uint8
        mask: Binary foreground mask, shape (H, W), uint8 {0, 255}

    Returns:
        Cropped BGR image containing only the foreground object, shape (h, w, 3), uint8
        Returns empty array if no foreground found.
    """
    # Ensure mask is binary
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Find bounding box of foreground
    coords = cv2.findNonZero(mask_binary)
    if coords is None:
        # No foreground - return empty
        return np.zeros((1, 1, 3), dtype=np.uint8)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop image and mask to bounding box
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = mask_binary[y:y+h, x:x+w]
    
    # Apply mask to cropped image (set background to black, then we'll crop it out)
    # Actually, since we're cropping to the bounding box, there shouldn't be background
    # But to be safe, apply the mask
    foreground_only = cropped_image * cropped_mask[:, :, np.newaxis]
    
    return foreground_only.astype(np.uint8)


def crop_to_foreground(
    image: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 3,
    padding: int = 0,
    alpha: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return ONLY the foreground: cropped to its bounding box, with a transparent background.

    Unlike extract_foreground_only (BGR, background inside the box becomes black), the pixels
    outside the object are fully transparent (alpha=0), so no background remains.

    Args:
        image: Original BGR image, shape (H, W, 3), uint8
        mask: Binary foreground mask, shape (H, W), uint8 {0, 255}
        feather_radius: Feather radius for smooth alpha edges (0 or 1 to disable)
        padding: Extra pixels kept around the bounding box
        alpha: Optional soft alpha matte (H, W) uint8, e.g. from the AI model. When given it is
            used directly instead of feathering the binary mask (better hair/edge detail).

    Returns:
        BGRA image, shape (h, w, 4), uint8. 1x1 fully transparent pixel if no foreground.
    """
    mask_binary = (mask > 0).astype(np.uint8)
    coords = cv2.findNonZero(mask_binary)
    if coords is None:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    if alpha is None:
        alpha = feather_edges(mask, radius=feather_radius)
    else:
        if alpha.shape[:2] != mask_binary.shape:
            raise ValueError("alpha and mask must have the same height/width")
        alpha = alpha.astype(np.uint8)
    # Colour is zeroed where alpha is 0 so no background pixels survive in the file.
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    bgra[alpha == 0, :3] = 0

    x, y, w, h = cv2.boundingRect(coords)
    H, W = mask_binary.shape
    x0, y0 = max(x - padding, 0), max(y - padding, 0)
    x1, y1 = min(x + w + padding, W), min(y + h + padding, H)
    return bgra[y0:y1, x0:x1].copy()


def composite_on_background(
    bgra: np.ndarray,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """Flatten a BGRA foreground onto a plain colour (for display in grids/previews)."""
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    bg = np.empty_like(bgra[:, :, :3], dtype=np.float32)
    bg[:] = background_color
    out = bgra[:, :, :3].astype(np.float32) * alpha + bg * (1.0 - alpha)
    return out.astype(np.uint8)


def extract_foreground(
    image: np.ndarray,
    mask: np.ndarray,
    feather_radius: int = 3
) -> Dict[str, np.ndarray]:
    """
    Produce all output formats from image and mask.

    Args:
        image: Original BGR image
        mask: Binary foreground mask
        feather_radius: Feather radius for transparent PNG (0 or 1 to disable)

    Returns:
        Dict with keys:
        - "transparent": RGBA image with transparency (feathered)
        - "black_bg": BGR image with black background (hard mask)
        - "binary_mask": BGR visualization of mask (hard mask)
        - "foreground_only": Cropped BGR image with ONLY foreground (no background)
    """
    transparent = create_transparent_png(image, mask, feather_radius=feather_radius)
    black_bg = apply_mask_to_image(image, mask, background_color=(0, 0, 0))
    binary_mask = create_binary_mask_image(mask)
    foreground_only = extract_foreground_only(image, mask)

    return {
        "transparent": transparent,
        "black_bg": black_bg,
        "binary_mask": binary_mask,
        "foreground_only": foreground_only
    }


def save_outputs(
    outputs: Dict[str, np.ndarray],
    base_path: str,
    prefix: str = ""
) -> Dict[str, str]:
    """
    Save all output images to disk.

    Args:
        outputs: Dict from extract_foreground
        base_path: Output directory
        prefix: Optional filename prefix

    Returns:
        Dict mapping output type to saved file path
    """
    os.makedirs(base_path, exist_ok=True)

    saved = {}
    for key, img in outputs.items():
        if key == "transparent":
            # Save as PNG to preserve alpha
            filename = f"{prefix}{key}.png" if prefix else f"{key}.png"
        else:
            filename = f"{prefix}{key}.png" if prefix else f"{key}.png"

        filepath = os.path.join(base_path, filename)
        cv2.imwrite(filepath, img)
        saved[key] = filepath

    return saved