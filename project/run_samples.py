"""
Batch script to run the extraction pipeline on all sample images.

Creates comparison grids (Original | Threshold Mask | GrabCut Mask | Final Cutout)
for the project report.
"""
import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.threshold import threshold_pipeline
from src.grabcut_seg import grabcut_pipeline
from src.extract import (
    extract_foreground, crop_to_foreground, composite_on_background,
    create_transparent_from_alpha,
)
from src import ai_segment
from src.utils import (
    load_image, resize_if_needed, scale_mask_to_original, 
    create_comparison_grid, ensure_bgr
)


def process_sample(image_path: Path, output_dir: Path, max_dim: int = 1024, mode: str = "classical") -> dict:
    """
    Process a single sample image through the full pipeline.
    
    Returns:
        Dictionary with paths to saved comparison images
    """
    try:
        # Load image
        image = load_image(str(image_path))
        image = ensure_bgr(image)
        
        # Resize for processing
        processed, scale = resize_if_needed(image, max_dimension=max_dim)
        
        alpha = None
        if mode == "ai":
            # AI: neural-network alpha matte at original resolution (no threshold / GrabCut)
            alpha, final_mask = ai_segment.ai_pipeline(image, max_dimension=max_dim)
            outputs = extract_foreground(image, final_mask)
            outputs["transparent"] = create_transparent_from_alpha(image, alpha)
            display_orig = cv2.resize(image, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_AREA) if scale > 1.0 else image
            thresh_vis = ai_segment.alpha_to_bgr(alpha)     # shown as the 2nd panel
            grabcut_vis = None
            grabcut_mask = None
            thresh_mask = None
        else:
            # Step 1: Threshold
            thresh_mask = threshold_pipeline(processed, method="otsu")

            # Step 2: GrabCut (with default rectangle)
            h, w = processed.shape[:2]
            default_rect = (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))
            grabcut_mask = grabcut_pipeline(processed, thresh_mask, rect=default_rect, iterations=5)

            # Step 3: Scale mask to original resolution
            if scale > 1.0:
                final_mask = scale_mask_to_original(grabcut_mask, image.shape[:2])
            else:
                final_mask = grabcut_mask

            # Step 4: Extract foreground
            outputs = extract_foreground(image, final_mask)

            # Visualizations for the comparison grid
            display_orig = cv2.resize(image, (processed.shape[1], processed.shape[0]), interpolation=cv2.INTER_AREA) if scale > 1.0 else image
            thresh_vis = cv2.cvtColor(thresh_mask, cv2.COLOR_GRAY2BGR)
            grabcut_vis = cv2.cvtColor(grabcut_mask, cv2.COLOR_GRAY2BGR)

        # Final image: ONLY the foreground (tight crop, transparent background).
        # Shown on white in the grid so no original background remains.
        fg_final = crop_to_foreground(image, final_mask, alpha=alpha)
        cutout_vis = composite_on_background(fg_final, (255, 255, 255))

        # Create comparison grid
        if mode == "ai":
            panels = [display_orig, thresh_vis, cutout_vis]
            titles = ["Original", "AI Alpha Matte", "Final (Foreground Only)"]
        else:
            panels = [display_orig, thresh_vis, grabcut_vis, cutout_vis]
            titles = ["Original", "Threshold Mask", "GrabCut Mask", "Final (Foreground Only)"]
        grid = create_comparison_grid(panels, titles, cols=len(panels), figsize=(4 * len(panels), 4))

        # Save comparison grid
        sample_name = image_path.stem
        grid_path = output_dir / f"{sample_name}_comparison.png"
        cv2.imwrite(str(grid_path), grid)
        
        # Also save individual outputs
        individual_dir = output_dir / "individual" / sample_name
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(individual_dir / "original.png"), image)
        if mode == "ai":
            cv2.imwrite(str(individual_dir / "ai_alpha_matte.png"), alpha)
        else:
            cv2.imwrite(str(individual_dir / "threshold_mask.png"), thresh_mask)
            cv2.imwrite(str(individual_dir / "grabcut_mask.png"), grabcut_mask)
        cv2.imwrite(str(individual_dir / "final_mask.png"), final_mask)
        cv2.imwrite(str(individual_dir / "transparent.png"), outputs["transparent"])
        cv2.imwrite(str(individual_dir / "black_bg.png"), outputs["black_bg"])
        cv2.imwrite(str(individual_dir / "binary_mask.png"), outputs["binary_mask"])
        cv2.imwrite(str(individual_dir / "foreground_only.png"), fg_final)  # cropped, transparent bg
        
        return {
            "sample": sample_name,
            "grid": str(grid_path),
            "individual_dir": str(individual_dir),
            "success": True
        }
        
    except Exception as e:
        return {
            "sample": image_path.stem,
            "error": str(e),
            "success": False
        }


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(description="Run the extraction pipeline on all samples.")
    parser.add_argument("--mode", choices=["classical", "ai"], default="classical",
                        help="classical = Threshold + GrabCut (default); ai = neural network (rembg)")
    parser.add_argument("--samples", default=None, help="Folder of images (default: ./samples)")
    args = parser.parse_args()
    mode = args.mode
    if mode == "ai" and not ai_segment.is_available():
        sys.exit("AI mode needs rembg: pip install rembg onnxruntime")

    project_root = Path(__file__).parent
    samples_dir = Path(args.samples) if args.samples else project_root / "samples"
    suffix = "_ai" if mode == "ai" else ""   # keep classical results untouched
    if args.samples:
        suffix += "_custom"                  # don't overwrite the built-in sample results
    output_dir = project_root / "outputs" / f"batch_results{suffix}"
    report_dir = project_root / "report" / f"result_comparison{suffix}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all sample images (excluding ground truth)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    sample_files = sorted([
        f for f in samples_dir.iterdir()
        if f.is_file() and f.suffix.lower() in exts and not f.stem.endswith("_mask")
    ])
    
    print(f"Mode: {mode} | Found {len(sample_files)} sample images")
    print("=" * 60)
    
    results = []
    
    for sample_path in sample_files:
        print(f"Processing {sample_path.name}...")
        result = process_sample(sample_path, output_dir, mode=mode)
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ Saved comparison: {result['grid']}")
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Create summary grid with all samples
    print("\nCreating summary comparison...")
    summary_images = []
    summary_titles = []
    
    for result in results:
        if result["success"]:
            # Load the individual images for a mini-grid
            indiv_dir = Path(result["individual_dir"])
            orig = cv2.imread(str(indiv_dir / "original.png"))
            if orig is not None:
                summary_images.append(cv2.resize(orig, (200, 200)))
                summary_titles.append(result["sample"])
    
    if summary_images:
        # Create a larger grid with all samples
        grid = create_comparison_grid(
            summary_images,
            summary_titles,
            cols=5,
            figsize=(20, 8)
        )
        summary_path = report_dir / "all_samples_summary.png"
        cv2.imwrite(str(summary_path), grid)
        print(f"Summary grid saved: {summary_path}")
    
    # Copy comparison grids to report directory
    import shutil
    for result in results:
        if result["success"]:
            src = Path(result["grid"])
            dst = report_dir / src.name
            shutil.copy2(src, dst)
    
    print(f"\nAll comparison grids copied to: {report_dir}")
    print("=" * 60)
    print(f"Processed: {sum(1 for r in results if r['success'])}/{len(results)} images")


if __name__ == "__main__":
    main()