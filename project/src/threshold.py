"""
Thresholding module: grayscale conversion, Gaussian blur, and binary thresholding.
"""
import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert input image (BGR or BGRA) to grayscale.

    Args:
        image: Input image as uint8 array, shape (H, W, 3) or (H, W, 4)

    Returns:
        Grayscale image as uint8 array, shape (H, W)

    Raises:
        ValueError: If image has unsupported number of channels
    """
    validate_image(image)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")


def apply_gaussian_blur(gray: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    """
    Apply Gaussian blur to grayscale image.

    Args:
        gray: Grayscale image, shape (H, W), uint8
        ksize: Kernel size (must be odd, >= 3)
        sigma: Gaussian standard deviation (0 = auto from ksize)

    Returns:
        Blurred grayscale image, shape (H, W), uint8
    """
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError("ksize must be odd and >= 3")
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


def compute_threshold_mask(
    gray: np.ndarray,
    method: str = "otsu",
    adaptive_block_size: int = 11,
    adaptive_c: int = 2
) -> np.ndarray:
    """
    Compute binary mask using thresholding.

    Args:
        gray: Grayscale image, shape (H, W), uint8
        method: "otsu" | "adaptive_mean" | "adaptive_gaussian"
        adaptive_block_size: Block size for adaptive threshold (odd, >= 3)
        adaptive_c: Constant subtracted from mean/weighted mean

    Returns:
        Binary mask as uint8 array, shape (H, W), values {0, 255}

    Raises:
        ValueError: If method is unknown
    """
    if method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
    elif method == "adaptive_mean":
        if adaptive_block_size % 2 == 0 or adaptive_block_size < 3:
            raise ValueError("adaptive_block_size must be odd and >= 3")
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            adaptive_block_size, adaptive_c
        )
    elif method == "adaptive_gaussian":
        if adaptive_block_size % 2 == 0 or adaptive_block_size < 3:
            raise ValueError("adaptive_block_size must be odd and >= 3")
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            adaptive_block_size, adaptive_c
        )
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def threshold_pipeline(
    image: np.ndarray,
    method: str = "otsu",
    blur_ksize: int = 5,
    adaptive_block_size: int = 11,
    adaptive_c: int = 2
) -> np.ndarray:
    """
    Complete thresholding pipeline: grayscale -> blur -> threshold.

    Args:
        image: Input BGR/BGRA image, uint8
        method: Thresholding method
        blur_ksize: Gaussian kernel size
        adaptive_block_size, adaptive_c: Adaptive threshold params

    Returns:
        Binary mask (0=background, 255=foreground), shape (H, W), uint8
    """
    gray = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(gray, ksize=blur_ksize)
    mask = compute_threshold_mask(
        blurred, method=method,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c
    )
    return mask


def validate_image(image: np.ndarray) -> tuple:
    """Validate image format and return (H, W, C)."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    if image.dtype != np.uint8:
        raise ValueError(f"Image must be uint8, got {image.dtype}")
    if image.ndim != 3:
        raise ValueError(f"Image must be 3D (H, W, C), got {image.ndim}D")
    if image.shape[2] not in (3, 4):
        raise ValueError(f"Image must have 3 or 4 channels, got {image.shape[2]}")
    return image.shape