# Foreground Object Extraction Tool - Design Document

**Project:** Foreground Object Extraction Tool  
**Version:** 1.0  
**Date:** 2026-10-28  
**Author:** Architect Agent  

---

## 1. Folder Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── threshold.py
│   ├── grabcut_seg.py
│   ├── extract.py
│   └── utils.py
├── app.py                      # Streamlit UI entry point
├── run_samples.py              # Batch processing script for report
├── requirements.txt
├── README.md
├── tests/
│   ├── __init__.py
│   ├── test_threshold.py
│   ├── test_grabcut_seg.py
│   ├── test_extract.py
│   └── test_utils.py
├── samples/                    # Input images for demo/batch
│   └── .gitkeep
├── outputs/                    # Generated outputs (gitignored)
│   └── .gitkeep
└── report/
    ├── report.md               # Final project report
    ├── flowchart/
    │   └── pipeline_flowchart.png
    └── result_comparison/      # Comparison grids from run_samples.py
        └── .gitkeep
```

---

## 2. Module Interfaces

### 2.1 `src/threshold.py`

**Purpose:** Apply grayscale conversion, Gaussian blur, and thresholding (Otsu or Adaptive) to produce an initial rough binary mask.

#### Public Functions

```python
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
    pass


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
    pass


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
    pass


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
    pass
```

---

### 2.2 `src/grabcut_seg.py`

**Purpose:** Run GrabCut segmentation initialized from threshold mask, with rectangle fallback.

#### Public Functions

```python
def create_grabcut_mask_from_threshold(
    thresh_mask: np.ndarray,
    prob_bg: int = cv2.GC_BGD,
    prob_fg: int = cv2.GC_FGD,
    pr_bg: int = cv2.GC_PR_BGD,
    pr_fg: int = cv2.GC_PR_FGD
) -> np.ndarray:
    """
    Convert binary threshold mask (0/255) to GrabCut mask (GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD).
    
    Strategy:
    - Pixels with value 255 in thresh_mask -> GC_PR_FGD (probable foreground)
    - Pixels with value 0 in thresh_mask -> GC_PR_BGD (probable background)
    - Optional: erode foreground to create GC_FGD core, dilate background for GC_BGD
    
    Args:
        thresh_mask: Binary mask from thresholding, values {0, 255}
        prob_bg, prob_fg, pr_bg, pr_fg: GrabCut mask values (default OpenCV constants)
        
    Returns:
        GrabCut initialization mask, shape (H, W), dtype=np.uint8
    """
    pass


def create_grabcut_mask_from_rect(
    image_shape: Tuple[int, int],
    rect: Tuple[int, int, int, int],  # (x, y, w, h)
    prob_bg: int = cv2.GC_BGD,
    prob_fg: int = cv2.GC_FGD,
    pr_bg: int = cv2.GC_PR_BGD,
    pr_fg: int = cv2.GC_PR_FGD
) -> np.ndarray:
    """
    Create GrabCut mask initialized from user-drawn rectangle.
    
    Inside rect -> GC_PR_FGD, outside -> GC_PR_BGD
    Optional: shrink rect for GC_FGD core
    
    Args:
        image_shape: (H, W) of image
        rect: (x, y, w, h) rectangle coordinates
        prob_bg, prob_fg, pr_bg, pr_fg: GrabCut mask values
        
    Returns:
        GrabCut initialization mask, shape (H, W), dtype=np.uint8
    """
    pass


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
    pass


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
    pass


def grabcut_pipeline(
    image: np.ndarray,
    thresh_mask: np.ndarray,
    rect: Optional[Tuple[int, int, int, int]] = None,
    iterations: int = 5,
    use_rect_fallback: bool = True
) -> np.ndarray:
    """
    Complete GrabCut pipeline with threshold mask primary, rect fallback.
    
    Args:
        image: Input BGR image
        thresh_mask: Binary mask from thresholding
        rect: Optional user rectangle (x, y, w, h)
        iterations: GrabCut iterations
        use_rect_fallback: If True and threshold mask fails (e.g., empty), use rect
        
    Returns:
        Binary foreground mask, shape (H, W), uint8 {0, 255}
    """
    pass
```

---

### 2.3 `src/extract.py`

**Purpose:** Apply binary mask to original image to produce three output formats.

#### Public Functions

```python
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
    pass


def create_transparent_png(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Create RGBA image with transparent background from mask.
    
    Args:
        image: Input BGR image, shape (H, W, 3), uint8
        mask: Binary mask, shape (H, W), uint8 {0, 255}
        
    Returns:
        RGBA image, shape (H, W, 4), uint8 (alpha=0 for background, 255 for foreground)
    """
    pass


def create_binary_mask_image(mask: np.ndarray) -> np.ndarray:
    """
    Create visualization of binary mask (white foreground on black background).
    
    Args:
        mask: Binary mask, shape (H, W), uint8 {0, 255}
        
    Returns:
        BGR visualization image, shape (H, W, 3), uint8
    """
    pass


def extract_foreground(
    image: np.ndarray,
    mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Produce all three output formats from image and mask.
    
    Args:
        image: Original BGR image
        mask: Binary foreground mask
        
    Returns:
        Dict with keys:
        - "transparent": RGBA image with transparency
        - "black_bg": BGR image with black background
        - "binary_mask": BGR visualization of mask
    """
    pass


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
    pass
```

---

### 2.4 `src/utils.py`

**Purpose:** Shared utilities for image I/O, validation, resizing, and visualization.

#### Public Functions

```python
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
    pass


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
    pass


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
    pass


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
    pass


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
    pass


def image_to_bytes(image: np.ndarray, format: str = ".png") -> bytes:
    """
    Encode image to bytes for Streamlit download buttons.
    
    Args:
        image: BGR or RGBA image
        format: Output format (".png", ".jpg")
        
    Returns:
        Encoded image bytes
    """
    pass


def bytes_to_image(data: bytes) -> np.ndarray:
    """
    Decode uploaded file bytes to BGR image.
    
    Args:
        data: Raw bytes from Streamlit file_uploader
        
    Returns:
        BGR image, uint8
    """
    pass
```

---

## 3. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE DATA FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

USER INPUT (Streamlit)
        │
        ▼
┌───────────────────┐
│  utils.load_image │  ──►  BGR uint8 image (H, W, 3)
│  utils.validate   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐     (optional, for large images)
│ utils.resize_if_  │  ──►  resized_image, scale_factor
│ needed()          │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ threshold_pipeline│  ──►  thresh_mask (H, W) uint8 {0, 255}
│ (threshold.py)    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ grabcut_pipeline  │  ──►  grabcut_mask (H, W) uint8 {0, 255}
│ (grabcut_seg.py)  │       (uses thresh_mask + optional rect)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐     (if resized earlier)
│ utils.scale_mask_ │  ──►  final_mask at original resolution
│ to_original()     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ extract_foreground│  ──►  {transparent, black_bg, binary_mask}
│ (extract.py)      │
└─────────┬─────────┘
          │
          ▼
    STREAMLIT UI
    - Display side-by-side
    - Download buttons
    - Save to outputs/
```

**Key Data Types:**
- All images: `np.ndarray` dtype `uint8`
- Masks: Single channel `(H, W)` values `{0, 255}`
- BGR images: `(H, W, 3)`
- RGBA images: `(H, W, 4)`
- Rectangles: `(x, y, w, h)` integers

---

## 4. UI Integration Points

### Streamlit App (`app.py`) Requirements from Each Module

| Module | UI Needs |
|--------|----------|
| **utils** | `load_image()`, `bytes_to_image()`, `image_to_bytes()`, `resize_if_needed()`, `create_comparison_grid()` |
| **threshold** | `threshold_pipeline()` with params: `method`, `blur_ksize`, `adaptive_block_size`, `adaptive_c` |
| **grabcut_seg** | `grabcut_pipeline()` with params: `iterations`, `rect` (from canvas), `use_rect_fallback` |
| **extract** | `extract_foreground()`, `save_outputs()` |

### UI Controls → Module Parameters

| UI Control | Module | Parameter |
|------------|--------|-----------|
| Threshold method dropdown | threshold | `method` |
| Blur kernel slider (odd, 3-31) | threshold | `blur_ksize` |
| Adaptive block size slider | threshold | `adaptive_block_size` |
| Adaptive C slider | threshold | `adaptive_c` |
| Rectangle selector (canvas) | grabcut_seg | `rect` |
| GrabCut iterations slider (1-20) | grabcut_seg | `iterations` |
| Use rect fallback checkbox | grabcut_seg | `use_rect_fallback` |

### Session State Management

```python
# Recommended session state keys
st.session_state.original_image      # Original BGR image (full res)
st.session_state.processed_image     # Resized image for processing
st.session_state.scale_factor        # Float
st.session_state.thresh_mask         # Threshold mask (processed res)
st.session_state.grabcut_mask        # GrabCut mask (processed res)
st.session_state.final_mask          # Final mask (original res)
st.session_state.outputs             # Dict from extract_foreground
st.session_state.rect                # (x, y, w, h) from canvas
```

---

## 5. Error Handling Strategy

### 5.1 Input Image Issues

| Scenario | Detection | Handling |
|----------|-----------|----------|
| **Grayscale input (1 channel)** | `validate_image()` returns C=1 | Convert to BGR by stacking: `cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)` |
| **RGBA input (4 channels)** | `validate_image()` returns C=4 | Split alpha; use RGB channels for processing; preserve alpha for transparent output |
| **Non-uint8 dtype** | `validate_image()` checks dtype | Convert: `image.astype(np.uint8)` with warning |
| **Corrupted/unreadable file** | `load_image()` catches cv2 error | Raise `ValueError` with user-friendly message |

### 5.2 Large Images

| Scenario | Detection | Handling |
|----------|-----------|----------|
| **Image > 1024px on either dimension** | `resize_if_needed()` with `max_dimension=1024` | Downscale for processing; store `scale_factor`; upscale final mask with `INTER_NEAREST` |
| **Memory pressure** | Check `image.nbytes > 50MB` | Force resize; warn user in UI |

### 5.3 Empty/Invalid Masks

| Scenario | Detection | Handling |
|----------|-----------|----------|
| **Threshold mask all zeros** | `np.count_nonzero(thresh_mask) == 0` | If `use_rect_fallback=True` and rect provided → use rect; else → return empty mask, warn user |
| **Threshold mask all 255** | `np.count_nonzero(thresh_mask) == H*W` | Treat as "no background found"; use rect fallback or return full image |
| **GrabCut fails to converge** | Check `bgd_model`/`fgd_model` validity | Return best-effort mask; log warning |
| **Rect outside image bounds** | `rect` validation | Clamp rect to image bounds; warn if significantly clamped |

### 5.4 Error Propagation

```python
# All public functions should:
# 1. Validate inputs early (fail fast)
# 2. Raise ValueError for invalid parameters
# 3. Return structured results (dicts/tuples), not raise on expected edge cases
# 4. Log warnings via Python logging module (not print)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests (pytest)

#### `tests/test_threshold.py`
```python
def test_convert_to_grayscale_bgr():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = convert_to_grayscale(img)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8

def test_convert_to_grayscale_bgra():
    img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    gray = convert_to_grayscale(img)
    assert gray.shape == (100, 100)

def test_convert_to_grayscale_invalid():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # 1 channel
    with pytest.raises(ValueError):
        convert_to_grayscale(img)

def test_apply_gaussian_blur():
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    blur = apply_gaussian_blur(gray, ksize=5)
    assert blur.shape == gray.shape

def test_compute_threshold_mask_otsu():
    # Create image with clear foreground/background
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[20:80, 20:80] = 255  # White square
    mask = compute_threshold_mask(gray, method="otsu")
    assert mask.shape == (100, 100)
    assert set(np.unique(mask)).issubset({0, 255})

def test_compute_threshold_mask_adaptive():
    gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    mask = compute_threshold_mask(gray, method="adaptive_mean")
    assert set(np.unique(mask)).issubset({0, 255})

def test_threshold_pipeline_end_to_end():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = threshold_pipeline(img, method="otsu")
    assert mask.shape == (100, 100)
    assert set(np.unique(mask)).issubset({0, 255})
```

#### `tests/test_grabcut_seg.py`
```python
def test_create_grabcut_mask_from_threshold():
    thresh = np.zeros((100, 100), dtype=np.uint8)
    thresh[20:80, 20:80] = 255
    gc_mask = create_grabcut_mask_from_threshold(thresh)
    assert gc_mask.shape == (100, 100)
    assert gc_mask.dtype == np.uint8
    # Check values are valid GC constants
    assert set(np.unique(gc_mask)).issubset({0, 1, 2, 3})  # GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD

def test_create_grabcut_mask_from_rect():
    gc_mask = create_grabcut_mask_from_rect((100, 100), (20, 20, 60, 60))
    assert gc_mask.shape == (100, 100)

def test_grabcut_to_binary_mask():
    gc_mask = np.full((50, 50), cv2.GC_PR_FGD, dtype=np.uint8)
    gc_mask[0:10, 0:10] = cv2.GC_BGD
    binary = grabcut_to_binary_mask(gc_mask)
    assert set(np.unique(binary)).issubset({0, 255})
    assert np.sum(binary == 255) == 50*50 - 10*10

def test_grabcut_pipeline_with_rect_fallback():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Empty threshold mask
    thresh = np.zeros((100, 100), dtype=np.uint8)
    mask = grabcut_pipeline(img, thresh, rect=(20, 20, 60, 60), use_rect_fallback=True)
    assert mask.shape == (100, 100)
    assert np.any(mask == 255)  # Should have foreground

def test_grabcut_pipeline_empty_no_fallback():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    thresh = np.zeros((100, 100), dtype=np.uint8)
    mask = grabcut_pipeline(img, thresh, use_rect_fallback=False)
    assert np.all(mask == 0)  # Empty result
```

#### `tests/test_extract.py`
```python
def test_apply_mask_to_image():
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    result = apply_mask_to_image(img, mask, background_color=(0, 0, 0))
    assert result.shape == (50, 50, 3)
    # Foreground preserved
    assert np.array_equal(result[20, 20], [128, 128, 128])
    # Background replaced
    assert np.array_equal(result[5, 5], [0, 0, 0])

def test_create_transparent_png():
    img = np.full((50, 50, 3), 255, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    rgba = create_transparent_png(img, mask)
    assert rgba.shape == (50, 50, 4)
    assert rgba[20, 20, 3] == 255  # Foreground opaque
    assert rgba[5, 5, 3] == 0      # Background transparent

def test_extract_foreground_returns_all():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    outputs = extract_foreground(img, mask)
    assert set(outputs.keys()) == {"transparent", "black_bg", "binary_mask"}
    assert outputs["transparent"].shape == (50, 50, 4)
    assert outputs["black_bg"].shape == (50, 50, 3)
    assert outputs["binary_mask"].shape == (50, 50, 3)
```

#### `tests/test_utils.py`
```python
def test_resize_if_needed_no_resize():
    img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    resized, scale = resize_if_needed(img, max_dimension=1024)
    assert scale == 1.0
    assert resized.shape == img.shape

def test_resize_if_needed_downscale():
    img = np.random.randint(0, 255, (2000, 1500, 3), dtype=np.uint8)
    resized, scale = resize_if_needed(img, max_dimension=1024)
    assert scale > 1.0
    assert max(resized.shape[:2]) <= 1024

def test_scale_mask_to_original():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    original_shape = (200, 200)
    scaled = scale_mask_to_original(mask, original_shape)
    assert scaled.shape == (200, 200)
    assert set(np.unique(scaled)).issubset({0, 255})

def test_image_to_bytes():
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    data = image_to_bytes(img, ".png")
    assert isinstance(data, bytes)
    assert len(data) > 0
    # Round-trip
    decoded = bytes_to_image(data)
    assert decoded.shape == img.shape

def test_create_comparison_grid():
    images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
    titles = ["A", "B", "C", "D"]
    grid = create_comparison_grid(images, titles, cols=4)
    assert grid.shape[0] == 100  # Height preserved
    assert grid.shape[2] == 3
```

### 6.2 Integration Tests

```python
# tests/test_integration.py
def test_full_pipeline_synthetic():
    """Test with synthetic image: white square on black background."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:150, 50:150] = [255, 255, 255]  # White square
    
    thresh = threshold_pipeline(img, method="otsu")
    gc_mask = grabcut_pipeline(img, thresh, iterations=3)
    outputs = extract_foreground(img, gc_mask)
    
    # Foreground should be extracted
    assert np.any(outputs["transparent"][:, :, 3] == 255)
    assert np.any(outputs["black_bg"] != 0)

def test_full_pipeline_with_rect():
    """Test pipeline with user rectangle."""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    thresh = threshold_pipeline(img, method="otsu")
    gc_mask = grabcut_pipeline(img, thresh, rect=(50, 50, 100, 100))
    outputs = extract_foreground(img, gc_mask)
    assert outputs["transparent"].shape == (200, 200, 4)
```

### 6.3 Test Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np

@pytest.fixture
def synthetic_image():
    """White square on gradient background."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        img[:, i] = [i, i, i]  # Horizontal gradient
    img[60:140, 60:140] = [255, 255, 255]
    return img

@pytest.fixture
def sample_rect():
    return (60, 60, 80, 80)
```

### 6.4 Test Execution

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 7. Assumptions & Decisions

| # | Assumption | Rationale |
|---|------------|-----------|
| 1 | **Input images are BGR (OpenCV default)** | Streamlit file_uploader returns RGB; we convert to BGR in `utils.bytes_to_image()` |
| 2 | **Max processing dimension = 1024px** | Balances speed/quality; user can adjust via constant |
| 3 | **Threshold mask → GrabCut: 255→PR_FGD, 0→PR_BGD** | Conservative initialization; GrabCut refines |
| 4 | **Rectangle fallback only when threshold fails** | Threshold is automatic; rect requires user effort |
| 5 | **Three fixed outputs (transparent PNG, black BG, mask)** | Matches requirements; no configurable output formats |
| 6 | **No CLI** | Explicit requirement: Streamlit only |
| 7 | **Batch script processes `samples/*.{png,jpg,jpeg}`** | Standard image extensions |
| 8 | **GrabCut iterations default=5, max=20** | Typical convergence; UI slider bounds |
| 9 | **Gaussian blur kernel must be odd** | OpenCV requirement; UI slider enforces odd values |
| 10 | **Masks always uint8 {0, 255}** | Consistent with OpenCV GrabCut expectations |
| 11 | **Report comparison grid: Original | Threshold | GrabCut | Cutout** | 4-panel layout as specified |
| 12 | **`run_samples.py` saves to `report/result_comparison/`** | Separate from user `outputs/` |

---

## 8. Implementation Priority

1. **Core modules** (`threshold.py`, `grabcut_seg.py`, `extract.py`, `utils.py`)
2. **Unit tests** (parallel with core)
3. **Streamlit app** (`app.py`)
4. **Batch script** (`run_samples.py`)
5. **Report assets** (flowchart, comparison figures)
6. **README & requirements**

---

## 9. Requirements.txt (Preview)

```
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
streamlit>=1.28.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

---

*End of Design Document*