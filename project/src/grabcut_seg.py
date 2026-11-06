"""
GrabCut segmentation module: initialization from threshold mask or rectangle, and execution.
Includes mask cleanup utilities for post-processing GrabCut results.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def morphological_cleanup(
    mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 5
) -> np.ndarray:
    """
    Apply morphological opening followed by closing to clean up the mask.

    Opening removes small noise (spurious foreground pixels).
    Closing fills small holes (spurious background pixels in foreground).

    Args:
        mask: Binary mask, shape (H, W), uint8 {0, 255}
        open_kernel: Kernel size for morphological opening (odd, >= 3). Use 0 to skip.
        close_kernel: Kernel size for morphological closing (odd, >= 3). Use 0 to skip.

    Returns:
        Cleaned binary mask, shape (H, W), uint8 {0, 255}
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Ensure binary
    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Handle edge cases
    if np.all(mask_binary == 0) or np.all(mask_binary == 255):
        return mask_binary

    # Validate kernel sizes
    if open_kernel > 0:
        if open_kernel % 2 == 0 or open_kernel < 3:
            raise ValueError("open_kernel must be odd and >= 3")
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel_open)

    if close_kernel > 0:
        if close_kernel % 2 == 0 or close_kernel < 3:
            raise ValueError("close_kernel must be odd and >= 3")
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close)

    return mask_binary


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in the foreground mask.

    Useful for removing small disconnected noise regions.

    Args:
        mask: Binary mask, shape (H, W), uint8 {0, 255}

    Returns:
        Mask with only the largest connected component, shape (H, W), uint8 {0, 255}
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Handle edge cases
    if np.all(mask_binary == 0):
        return mask_binary
    if np.all(mask_binary == 255):
        return mask_binary

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)

    if num_labels <= 1:
        # Only background (label 0) exists
        return mask_binary

    # Find largest component (excluding background label 0)
    # stats[:, cv2.CC_STAT_AREA] gives area of each component
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
    if len(areas) == 0:
        return np.zeros_like(mask_binary)

    largest_label = np.argmax(areas) + 1  # +1 because we skipped background

    # Create mask with only largest component
    result = np.zeros_like(mask_binary)
    result[labels == largest_label] = 255

    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in the foreground mask.

    Uses flood fill from image borders to identify background,
    then inverts to get filled foreground.

    Args:
        mask: Binary mask, shape (H, W), uint8 {0, 255}

    Returns:
        Mask with holes filled, shape (H, W), uint8 {0, 255}
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Handle edge cases
    if np.all(mask_binary == 0) or np.all(mask_binary == 255):
        return mask_binary

    h, w = mask_binary.shape
    # Create a mask for floodFill that is 2 pixels larger
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Copy for flood fill
    flood_fill = mask_binary.copy()

    # Flood fill from top-left corner (0, 0) - must be background
    # If (0,0) is foreground, find a background pixel on border
    if flood_fill[0, 0] == 255:
        # Search border for background pixel
        border_bg = None
        # Top and bottom borders
        for x in range(w):
            if flood_fill[0, x] == 0:
                border_bg = (x, 0)
                break
            if flood_fill[h - 1, x] == 0:
                border_bg = (x, h - 1)
                break
        # Left and right borders
        if border_bg is None:
            for y in range(h):
                if flood_fill[y, 0] == 0:
                    border_bg = (0, y)
                    break
                if flood_fill[y, w - 1] == 0:
                    border_bg = (w - 1, y)
                    break
        if border_bg is None:
            # Entire border is foreground - no holes to fill
            return mask_binary
        cv2.floodFill(flood_fill, flood_mask, border_bg, 255)
    else:
        cv2.floodFill(flood_fill, flood_mask, (0, 0), 255)

    # Invert flood fill result to get background
    flood_fill_inv = cv2.bitwise_not(flood_fill)

    # Combine original mask with inverted flood fill (fills holes)
    filled = cv2.bitwise_or(mask_binary, flood_fill_inv)

    return filled


def cleanup_mask(
    mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 5,
    keep_largest: bool = True,
    fill_holes_flag: bool = True
) -> np.ndarray:
    """
    Combined mask cleanup pipeline.

    Applies in order:
    1. Morphological opening (noise removal)
    2. Morphological closing (hole filling)
    3. Fill holes (flood-fill based)
    4. Keep largest connected component

    Args:
        mask: Binary mask, shape (H, W), uint8 {0, 255}
        open_kernel: Kernel size for opening (0 to skip)
        close_kernel: Kernel size for closing (0 to skip)
        keep_largest: Whether to keep only largest connected component
        fill_holes_flag: Whether to apply flood-fill hole filling

    Returns:
        Cleaned binary mask, shape (H, W), uint8 {0, 255}
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask_binary = (mask > 0).astype(np.uint8) * 255

    # Early return for empty or full masks
    if np.all(mask_binary == 0) or np.all(mask_binary == 255):
        return mask_binary

    # Step 1 & 2: Morphological open/close
    mask_binary = morphological_cleanup(mask_binary, open_kernel, close_kernel)

    # Step 3: Fill holes
    if fill_holes_flag:
        mask_binary = fill_holes(mask_binary)

    # Step 4: Keep largest component
    if keep_largest:
        mask_binary = keep_largest_component(mask_binary)

    return mask_binary


def create_grabcut_mask_from_threshold(
    thresh_mask: np.ndarray,
    prob_bg: int = cv2.GC_BGD,
    prob_fg: int = cv2.GC_FGD,
    pr_bg: int = cv2.GC_PR_BGD,
    pr_fg: int = cv2.GC_PR_FGD,
    erode_kernel: int = 3
) -> np.ndarray:
    """
    Convert binary threshold mask (0/255) to GrabCut mask (GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD).

    Strategy:
    - Pixels with value 255 in thresh_mask -> GC_PR_FGD (probable foreground)
    - Pixels with value 0 in thresh_mask -> GC_PR_BGD (probable background)
    - Erode foreground to create GC_FGD core, dilate background for GC_BGD

    Args:
        thresh_mask: Binary mask from thresholding, values {0, 255}
        prob_bg, prob_fg, pr_bg, pr_fg: GrabCut mask values (default OpenCV constants)
        erode_kernel: Kernel size for erosion/dilation to create definite regions

    Returns:
        GrabCut initialization mask, shape (H, W), dtype=np.uint8
    """
    if thresh_mask.dtype != np.uint8:
        thresh_mask = thresh_mask.astype(np.uint8)

    h, w = thresh_mask.shape
    gc_mask = np.full((h, w), pr_bg, dtype=np.uint8)  # Default: probable background

    # Threshold foreground -> probable foreground
    fg_pixels = (thresh_mask == 255)
    gc_mask[fg_pixels] = pr_fg

    # Create definite foreground by eroding
    if erode_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel))
        fg_definite = cv2.erode(thresh_mask, kernel, iterations=1)
        gc_mask[fg_definite == 255] = prob_fg

        # Create definite background by dilating background
        bg_definite = cv2.dilate(thresh_mask, kernel, iterations=1)
        gc_mask[bg_definite == 0] = prob_bg

    return gc_mask


def create_grabcut_mask_from_rect(
    image_shape: Tuple[int, int],
    rect: Tuple[int, int, int, int],  # (x, y, w, h)
    prob_bg: int = cv2.GC_BGD,
    prob_fg: int = cv2.GC_FGD,
    pr_bg: int = cv2.GC_PR_BGD,
    pr_fg: int = cv2.GC_PR_FGD,
    shrink_factor: float = 0.1
) -> np.ndarray:
    """
    Create GrabCut mask initialized from user-drawn rectangle.

    Inside rect -> GC_PR_FGD, outside -> GC_PR_BGD
    Shrunken rect -> GC_FGD

    Args:
        image_shape: (H, W) of image
        rect: (x, y, w, h) rectangle coordinates
        prob_bg, prob_fg, pr_bg, pr_fg: GrabCut mask values
        shrink_factor: Fraction to shrink rect for definite foreground

    Returns:
        GrabCut initialization mask, shape (H, W), dtype=np.uint8
    """
    h, w = image_shape
    gc_mask = np.full((h, w), pr_bg, dtype=np.uint8)

    x, y, rw, rh = rect
    # Clamp rect to image bounds
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))

    # Probable foreground inside rect
    gc_mask[y:y+rh, x:x+rw] = pr_fg

    # Definite foreground: shrunken rect
    if shrink_factor > 0:
        sx = int(x + rw * shrink_factor)
        sy = int(y + rh * shrink_factor)
        srw = max(1, int(rw * (1 - 2 * shrink_factor)))
        srh = max(1, int(rh * (1 - 2 * shrink_factor)))
        if srw > 0 and srh > 0:
            gc_mask[sy:sy+srh, sx:sx+srw] = prob_fg

    return gc_mask


def run_grabcut(
    image: np.ndarray,
    init_mask: np.ndarray,
    iterations: int = 5,
    mode: int = cv2.GC_INIT_WITH_MASK
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run OpenCV GrabCut algorithm.

    Args:
        image: Input BGR image, shape (H, W, 3), uint8
        init_mask: Initialization mask from create_grabcut_mask_from_threshold/rect
        iterations: Number of GrabCut iterations
        mode: cv2.GC_INIT_WITH_MASK or cv2.GC_INIT_WITH_RECT

    Returns:
        Tuple of (final_mask, bgd_model, fgd_model)
        - final_mask: GrabCut result mask with GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD
        - bgd_model: Background GMM model (13*5 float64)
        - fgd_model: Foreground GMM model (13*5 float64)
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    h, w = image.shape[:2]
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    mask = init_mask.copy()
    cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, mode)

    return mask, bgd_model, fgd_model


def grabcut_to_binary_mask(grabcut_mask: np.ndarray) -> np.ndarray:
    """
    Convert GrabCut result mask to binary mask (0=bg, 255=fg).

    GC_FGD and GC_PR_FGD -> 255 (foreground)
    GC_BGD and GC_PR_BGD -> 0 (background)

    Args:
        grabcut_mask: Output from run_grabcut

    Returns:
        Binary mask, shape (H, W), uint8, values {0, 255}
    """
    binary = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)
    return binary


def grabcut_pipeline(
    image: np.ndarray,
    thresh_mask: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
    iterations: int = 5,
    use_rect_fallback: bool = True,
    cleanup: bool = True,
    cleanup_open_kernel: int = 3,
    cleanup_close_kernel: int = 5,
    cleanup_keep_largest: bool = True,
    cleanup_fill_holes: bool = True
) -> np.ndarray:
    """
    Complete GrabCut pipeline with threshold mask primary, rect fallback, and optional mask cleanup.

    Args:
        image: Input BGR image
        thresh_mask: Binary mask from thresholding
        rect: Optional user rectangle (x, y, w, h)
        iterations: GrabCut iterations
        use_rect_fallback: If True and threshold mask fails (e.g., empty), use rect
        cleanup: Whether to apply post-processing mask cleanup
        cleanup_open_kernel: Kernel size for morphological opening (0 to skip)
        cleanup_close_kernel: Kernel size for morphological closing (0 to skip)
        cleanup_keep_largest: Whether to keep only largest connected component
        cleanup_fill_holes: Whether to apply flood-fill hole filling

    Returns:
        Binary foreground mask, shape (H, W), uint8 {0, 255}
    """
    # Check if threshold mask is usable
    fg_pixels = np.count_nonzero(thresh_mask)
    total_pixels = thresh_mask.size
    fg_ratio = fg_pixels / total_pixels

    # Threshold mask is empty or nearly empty
    if fg_pixels == 0 or fg_ratio < 0.001:
        if use_rect_fallback and rect is not None:
            init_mask = create_grabcut_mask_from_rect(image.shape[:2], rect)
            mode = cv2.GC_INIT_WITH_MASK
        else:
            # Return empty mask
            return np.zeros(image.shape[:2], dtype=np.uint8)
    # Threshold mask is full or nearly full
    elif fg_ratio > 0.999:
        if use_rect_fallback and rect is not None:
            init_mask = create_grabcut_mask_from_rect(image.shape[:2], rect)
            mode = cv2.GC_INIT_WITH_MASK
        else:
            # Return full mask
            return np.full(image.shape[:2], 255, dtype=np.uint8)
    else:
        # Normal case: use threshold mask
        init_mask = create_grabcut_mask_from_threshold(thresh_mask)
        mode = cv2.GC_INIT_WITH_MASK

    # Run GrabCut
    final_mask, _, _ = run_grabcut(image, init_mask, iterations, mode)
    binary_mask = grabcut_to_binary_mask(final_mask)

    # Apply mask cleanup if requested
    if cleanup:
        binary_mask = cleanup_mask(
            binary_mask,
            open_kernel=cleanup_open_kernel,
            close_kernel=cleanup_close_kernel,
            keep_largest=cleanup_keep_largest,
            fill_holes_flag=cleanup_fill_holes
        )

    return binary_mask