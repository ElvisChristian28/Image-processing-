"""
Utility functions for image I/O, validation, resizing, and visualization.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io


def load_image(path: str) -> np.ndarray:
    """
    Load image from file, ensuring BGR uint8 format.

    Args:
        path: Path to image file

    Returns:
        BGR image, shape (H, W, 3) or (H, W, 4), uint8

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If image cannot be decoded
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    return image


def validate_image(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Validate image format and return (H, W, C).

    Args:
        image: Input array

    Returns:
        Tuple of (height, width, channels)

    Raises:
        ValueError: If image is not valid uint8 array with 3 or 4 channels
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError(f"Image must be uint8, got {image.dtype}")
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D (H, W, C), got {image.ndim}D")
    if image.shape[2] not in (3, 4):
        raise ValueError(f"Image must have 3 or 4 channels, got {image.shape[2]}")
    return image.shape


def resize_if_needed(
    image: np.ndarray,
    max_dimension: int = 1024,
    interpolation: int = cv2.INTER_AREA
) -> Tuple[np.ndarray, float]:
    """
    Resize image if larger than max_dimension, preserving aspect ratio.

    Args:
        image: Input image
        max_dimension: Maximum allowed width or height
        interpolation: OpenCV interpolation method

    Returns:
        Tuple of (resized_image, scale_factor)
        scale_factor = original_dim / resized_dim (>= 1.0)
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim <= max_dimension:
        return image, 1.0

    scale = max_dimension / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, 1.0 / scale  # scale_factor > 1 means original was larger


def scale_mask_to_original(
    mask: np.ndarray,
    original_shape: Tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST
) -> np.ndarray:
    """
    Scale mask back to original image dimensions.

    Args:
        mask: Mask at processed resolution
        original_shape: (H, W) of original image
        interpolation: cv2.INTER_NEAREST recommended for binary masks

    Returns:
        Mask at original resolution
    """
    h, w = original_shape
    if mask.shape[:2] == (h, w):
        return mask

    scaled = cv2.resize(mask, (w, h), interpolation=interpolation)
    # Ensure binary after scaling
    _, scaled = cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)
    return scaled


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    cols: int = 4,
    figsize: Tuple[int, int] = (16, 4)
) -> np.ndarray:
    """
    Create a horizontal comparison grid for visualization/report.

    Args:
        images: List of BGR/RGB images (same size)
        titles: List of titles for each image
        cols: Number of columns
        figsize: Matplotlib figure size

    Returns:
        Grid image as BGR uint8 array
    """
    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    # squeeze=False ensures axes is always 2D array

    for i, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        # Convert BGR to RGB for matplotlib
        if img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img_rgb = img

        ax.imshow(img_rgb)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis('off')

    plt.tight_layout()

    # Save to bytes and convert back to OpenCV
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)

    img_array = np.frombuffer(buf.read(), dtype=np.uint8)
    grid = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return grid


def image_to_bytes(image: np.ndarray, format: str = ".png") -> bytes:
    """
    Encode image to bytes for Streamlit download buttons.

    Args:
        image: BGR or RGBA image
        format: Output format (".png", ".jpg")

    Returns:
        Encoded image bytes
    """
    success, encoded = cv2.imencode(format, image)
    if not success:
        raise ValueError(f"Failed to encode image as {format}")
    return encoded.tobytes()


def bytes_to_image(data: bytes) -> np.ndarray:
    """
    Decode uploaded file bytes to BGR image.

    Args:
        data: Raw bytes from Streamlit file_uploader

    Returns:
        BGR image, uint8
    """
    nparr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode uploaded image")

    # Ensure 3 or 4 channels
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # Keep BGRA for processing, convert to BGR when needed
        pass

    return image


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Ensure image is 3-channel BGR.

    Args:
        image: Input image (BGR, BGRA, or grayscale)

    Returns:
        BGR image
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image