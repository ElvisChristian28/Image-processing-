# Foreground Object Extraction Tool

A Python tool for extracting foreground objects from images using thresholding and GrabCut segmentation. Built with OpenCV, NumPy, and Streamlit.

## Features

- **AI mode (new)** - Local neural-network background removal (rembg) for real photos; no API key, works offline
- **Automatic thresholding** - Otsu's method, Adaptive Mean, Adaptive Gaussian
- **GrabCut refinement** - Improves mask quality with user-guided or automatic initialization
- **Mask cleanup** - Morphological operations, hole filling, largest component selection
- **Edge feathering** - Smooth alpha transitions for professional cutouts
- **Multiple outputs** - Transparent PNG, black background, binary mask
- **Interactive UI** - Streamlit web interface with rectangle selector and real-time preview
- **Batch processing** - Run on multiple images for evaluation

## Installation

```bash
# Clone or navigate to the project directory
cd project

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- OpenCV (cv2)
- NumPy
- Matplotlib
- Streamlit
- streamlit-drawable-canvas
- *(optional, for AI mode)* rembg, onnxruntime, Pillow

## AI Mode (Background Removal for Real Photos)

Classical Threshold + GrabCut separates objects by brightness/colour, so it struggles on real
photos (e.g. dark hair in front of dark tree trunks). AI mode uses a pre-trained segmentation
network via [rembg](https://github.com/danielgatis/rembg) instead.

- **Local & free** - no API key, no account, photos never leave your computer.
- **First run** downloads the model once (~170 MB) and caches it; afterwards it works offline.
- **CPU only** - a typical photo takes a few seconds.
- **Soft alpha matte** - the transparent PNG keeps fine hair/edge detail (no hard cut-out edge).

| Model | Use for |
| --- | --- |
| People / portraits (`u2net_human_seg`) | photos of people |
| General objects (`isnet-general-use`) | products, animals, objects, graphics |

**In the app:** pick *AI (neural network)* in the sidebar *Mode* section (default when `rembg`
is installed). The rectangle canvas and threshold sliders are only needed in Classical mode.

**Batch:**

```bash
python run_samples.py                 # classical (default) -> outputs/batch_results
python run_samples.py --mode ai       # AI                  -> outputs/batch_results_ai
python run_samples.py --mode ai --samples path/to/photos   # your own images (png/jpg/webp/...)
```

**From code:**

```python
from src.ai_segment import ai_pipeline
from src.extract import crop_to_foreground

alpha, mask = ai_pipeline(bgr_image, model="people")          # soft matte + binary mask
fg = crop_to_foreground(bgr_image, mask, alpha=alpha)         # cropped BGRA, transparent bg
```

**Which mode when?** Measured on the 10 synthetic samples (mean IoU vs ground truth):
Classical 0.81, AI *general* 0.84, AI *people* 0.13 (the people model only recognises people,
so use it for portraits). On real photos Classical typically fails while AI works - on the test
portrait the Otsu threshold mask was unusable and GrabCut kept 100% of the frame.

## Usage

### Streamlit Web Interface

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

#### UI Controls

1. **Upload Image** - Support for PNG, JPG, JPEG, BMP, TIFF, WebP
2. **Threshold Settings**
   - Method: Otsu / Adaptive Mean / Adaptive Gaussian
   - Blur Kernel Size: Gaussian blur before thresholding (odd values 3-31)
   - Adaptive Block Size: Neighborhood size for adaptive methods (odd 3-51)
   - Adaptive C: Constant subtracted from mean (0-20)
3. **GrabCut Settings**
   - Iterations: Number of refinement iterations (1-20)
   - Use Rectangle Fallback: Use drawn rectangle if threshold fails
4. **Rectangle Selector** - Draw a rectangle around the foreground object on the canvas
5. **Actions** - Re-run pipeline, clear rectangle, reset all

#### Outputs (Download Buttons)

- **Transparent PNG** - RGBA with alpha channel for compositing
- **Black Background** - BGR image with black background for viewing
- **Binary Mask** - White foreground on black background for further processing

### Batch Processing (for Reports)

```bash
python run_samples.py
```

Processes all images in `samples/` and saves comparison grids to `outputs/batch_results/` and `report/result_comparison/`.

## Project Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── threshold.py       # Thresholding pipeline (grayscale → blur → threshold)
│   ├── grabcut_seg.py     # GrabCut segmentation with mask cleanup
│   ├── ai_segment.py      # AI background removal (rembg, local, optional)
│   ├── extract.py         # Foreground extraction, edge feathering, cropped transparent output
│   └── utils.py           # Image I/O, resizing, visualization utilities
├── app.py                 # Streamlit web application
├── run_samples.py         # Batch processing script
├── requirements.txt
├── samples/               # Test images
│   └── ground_truth/      # Ground truth masks for evaluation
├── outputs/               # Generated outputs (gitignored)
├── tests/
│   ├── test_threshold.py
│   ├── test_grabcut_seg.py
│   ├── test_extract.py
│   ├── test_ai_segment.py # AI module (network mocked) + optional real-model test
│   ├── test_utils.py
│   └── evaluate.py        # IoU and accuracy evaluation
└── report/
    ├── report.md          # Project report
    ├── flowchart/
    └── result_comparison/ # Comparison grids
```

## Algorithm Pipeline

```
Input Image
    │
    ▼
┌─────────────────┐
│  Grayscale      │  cv2.cvtColor
│  Conversion     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gaussian Blur  │  cv2.GaussianBlur
│  (noise reduce) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Thresholding   │  Otsu / Adaptive Mean / Adaptive Gaussian
│  (initial mask) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GrabCut Init   │  Threshold mask → GC_PR_FGD/PR_BGD
│  (from mask)    │  Erode → GC_FGD, Dilate → GC_BGD
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GrabCut Run    │  cv2.grabCut (GC_INIT_WITH_MASK)
│  (refinement)   │  User rect fallback (GC_INIT_WITH_RECT)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Mask Cleanup   │  Morphological open/close
│  (post-process) │  Fill holes (flood-fill)
                  │  Keep largest component
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extraction     │  Transparent PNG (feathered alpha)
│  (3 outputs)    │  Black background (hard mask)
                  │  Binary mask visualization
└────────┴────────┘
```

## API Reference

### `src.threshold`

```python
from src.threshold import threshold_pipeline

mask = threshold_pipeline(
    image,                    # BGR/BGRA image
    method="otsu",           # "otsu" | "adaptive_mean" | "adaptive_gaussian"
    blur_ksize=5,            # Gaussian kernel (odd, >=3)
    adaptive_block_size=11,  # For adaptive methods
    adaptive_c=2
)
# Returns: binary mask (H, W) uint8 {0, 255}
```

### `src.grabcut_seg`

```python
from src.grabcut_seg import grabcut_pipeline, cleanup_mask

mask = grabcut_pipeline(
    image,                    # BGR image
    thresh_mask,              # Binary mask from thresholding
    rect=(x, y, w, h),        # Optional user rectangle
    iterations=5,             # GrabCut iterations
    use_rect_fallback=True,   # Fallback to rect if threshold fails
    cleanup=True,             # Apply post-processing cleanup
    cleanup_open_kernel=3,    # Morph open kernel (0 to skip)
    cleanup_close_kernel=5,   # Morph close kernel (0 to skip)
    cleanup_keep_largest=True,
    cleanup_fill_holes=True
)

# Standalone cleanup
clean_mask = cleanup_mask(
    mask,
    open_kernel=3,
    close_kernel=5,
    keep_largest=True,
    fill_holes_flag=True
)
```

### `src.extract`

```python
from src.extract import extract_foreground, create_transparent_png, feather_edges

outputs = extract_foreground(
    image,           # BGR image
    mask,            # Binary mask
    feather_radius=3 # Edge feathering (0/1 to disable)
)
# Returns: {"transparent": RGBA, "black_bg": BGR, "binary_mask": BGR}
```

### `src.utils`

```python
from src.utils import (
    load_image, bytes_to_image, image_to_bytes,
    resize_if_needed, scale_mask_to_original,
    create_comparison_grid, ensure_bgr
)
```

## Evaluation

Run the evaluation script to compute IoU and pixel accuracy:

```bash
python tests/evaluate.py
```

This compares:
- Threshold-only vs Ground Truth
- Threshold + GrabCut vs Ground Truth

Results are saved to `report/evaluation/evaluation_results.json`.

## Example Results

| Sample | Threshold IoU | GrabCut IoU | Improvement |
|--------|--------------|-------------|-------------|
| Simple | 0.9998 | 0.9998 | +0.0000 |
| Cluttered | 0.7656 | 0.9942 | **+0.2286** |
| Low Contrast | 0.9998 | 0.9998 | +0.0000 |
| Person-like | 0.9997 | 0.9990 | -0.0007 |
| Small Object | 0.9874 | 1.0000 | +0.0126 |
| Translucent | 0.9998 | 1.0000 | +0.0002 |
| Multiple Objects | 0.5002 | 0.2715 | -0.2287 |
| Touching Border | 1.0000 | 0.6097 | -0.3902 |
| Thin Object | 0.7867 | 0.9906 | **+0.2039** |
| Gradient BG | 0.2762 | 0.2776 | +0.0015 |

GrabCut significantly improves results on cluttered backgrounds, thin objects, and small objects. For simple high-contrast scenes, threshold alone is sufficient. For multiple objects or border-touching objects, the default rectangle may not be optimal.

## Limitations & Future Work

### Limitations
- **Multiple objects**: Default rectangle encompasses all objects; GrabCut tends to merge them
- **Border-touching objects**: Rectangle initialization struggles when object touches image edge
- **Gradient backgrounds**: Low contrast gradients challenge both thresholding and GrabCut
- **Thin structures**: Very thin objects (< 3px) may be lost in morphological cleanup

### Future Improvements
- Interactive brush strokes for precise GrabCut initialization (GC_FGD/GC_BGD)
- Automatic rectangle detection using saliency or object detection
- Deep learning-based segmentation (SAM, DeepLab) as alternative backend
- Batch UI for processing multiple images
- Video frame extraction support

## License

MIT License - Free for personal and commercial use.

## Credits

- OpenCV GrabCut implementation (based on "GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts" by Rother et al.)
- Streamlit for the web framework
- streamlit-drawable-canvas for the rectangle selector