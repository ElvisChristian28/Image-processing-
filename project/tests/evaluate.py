"""
Evaluation script for Foreground Object Extraction Tool.

Computes IoU (Intersection over Union) and Pixel Accuracy against ground truth masks.
Compares threshold-only vs threshold+GrabCut results.
"""
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.threshold import threshold_pipeline
from src.grabcut_seg import grabcut_pipeline
from src.extract import extract_foreground
from src.utils import load_image, resize_if_needed, scale_mask_to_original, create_comparison_grid


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.
    
    Args:
        mask1: First binary mask (0 or 255)
        mask2: Second binary mask (0 or 255)
    
    Returns:
        IoU score in [0, 1]
    """
    # Ensure binary
    m1 = (mask1 > 127).astype(np.uint8)
    m2 = (mask2 > 127).astype(np.uint8)
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def compute_pixel_accuracy(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute pixel accuracy between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
    
    Returns:
        Accuracy in [0, 1]
    """
    m1 = (mask1 > 127).astype(np.uint8)
    m2 = (mask2 > 127).astype(np.uint8)
    
    correct = (m1 == m2).sum()
    total = m1.size
    
    return correct / total


def compute_precision_recall_f1(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.
    
    Returns:
        (precision, recall, f1)
    """
    m1 = (mask_pred > 127).astype(np.uint8)
    m2 = (mask_gt > 127).astype(np.uint8)
    
    tp = np.logical_and(m1 == 1, m2 == 1).sum()
    fp = np.logical_and(m1 == 1, m2 == 0).sum()
    fn = np.logical_and(m1 == 0, m2 == 1).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def run_threshold_only(image: np.ndarray) -> np.ndarray:
    """Run threshold-only pipeline (no GrabCut)."""
    thresh_mask = threshold_pipeline(image, method="otsu")
    return thresh_mask


def run_threshold_grabcut(image: np.ndarray, rect: Tuple[int, int, int, int] = None) -> np.ndarray:
    """Run threshold + GrabCut pipeline."""
    thresh_mask = threshold_pipeline(image, method="otsu")
    grabcut_mask = grabcut_pipeline(image, thresh_mask, rect=rect, iterations=5)
    return grabcut_mask


def process_sample(sample_name: str, samples_dir: Path, gt_dir: Path, output_dir: Path) -> Dict:
    """
    Process a single sample image and compute metrics.
    
    Returns:
        Dictionary with metrics for both methods
    """
    # Load image and ground truth
    img_path = samples_dir / f"{sample_name}.png"
    gt_path = gt_dir / f"{sample_name}_mask.png"
    
    if not img_path.exists() or not gt_path.exists():
        return {"error": f"Missing files for {sample_name}"}
    
    image = load_image(str(img_path))
    gt_mask = load_image(str(gt_path))
    
    # Ground truth might be 3-channel, convert to single channel
    if gt_mask.ndim == 3:
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
    
    # Ensure binary
    gt_mask = (gt_mask > 127).astype(np.uint8) * 255
    
    # Resize image for processing if needed
    processed_image, scale = resize_if_needed(image, max_dimension=1024)
    
    # Define a default rectangle (center of image, 60% of dimensions)
    h, w = processed_image.shape[:2]
    default_rect = (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))
    
    # Method 1: Threshold only
    thresh_mask_processed = run_threshold_only(processed_image)
    thresh_mask = scale_mask_to_original(thresh_mask_processed, image.shape[:2])
    
    # Method 2: Threshold + GrabCut
    gc_mask_processed = run_threshold_grabcut(processed_image, rect=default_rect)
    gc_mask = scale_mask_to_original(gc_mask_processed, image.shape[:2])
    
    # Compute metrics
    thresh_iou = compute_iou(thresh_mask, gt_mask)
    thresh_acc = compute_pixel_accuracy(thresh_mask, gt_mask)
    thresh_prec, thresh_rec, thresh_f1 = compute_precision_recall_f1(thresh_mask, gt_mask)
    
    gc_iou = compute_iou(gc_mask, gt_mask)
    gc_acc = compute_pixel_accuracy(gc_mask, gt_mask)
    gc_prec, gc_rec, gc_f1 = compute_precision_recall_f1(gc_mask, gt_mask)
    
    # Create comparison grid
    comparison_dir = output_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare visualizations
    thresh_vis = cv2.cvtColor(thresh_mask, cv2.COLOR_GRAY2BGR)
    gc_vis = cv2.cvtColor(gc_mask, cv2.COLOR_GRAY2BGR)
    gt_vis = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    
    # Extract foreground for visualization
    outputs = extract_foreground(image, gc_mask)
    cutout_vis = outputs["black_bg"]
    
    # Create comparison grid: Original | GT | Threshold | GrabCut | Cutout
    grid = create_comparison_grid(
        [image, gt_vis, thresh_vis, gc_vis, cutout_vis],
        ["Original", "Ground Truth", "Threshold", "GrabCut", "Cutout"],
        cols=5,
        figsize=(20, 4)
    )
    
    grid_path = comparison_dir / f"{sample_name}_comparison.png"
    cv2.imwrite(str(grid_path), grid)
    
    return {
        "sample": sample_name,
        "threshold": {
            "iou": thresh_iou,
            "accuracy": thresh_acc,
            "precision": thresh_prec,
            "recall": thresh_rec,
            "f1": thresh_f1
        },
        "grabcut": {
            "iou": gc_iou,
            "accuracy": gc_acc,
            "precision": gc_prec,
            "recall": gc_rec,
            "f1": gc_f1
        },
        "improvement": {
            "iou_delta": gc_iou - thresh_iou,
            "accuracy_delta": gc_acc - thresh_acc
        }
    }


def main():
    """Main evaluation function."""
    project_root = Path(__file__).parent.parent
    samples_dir = project_root / "samples"
    gt_dir = samples_dir / "ground_truth"
    output_dir = project_root / "outputs"
    report_dir = project_root / "report" / "evaluation"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all sample images
    sample_files = list(samples_dir.glob("*.png"))
    sample_names = [f.stem for f in sample_files if not f.stem.endswith("_mask")]
    
    print(f"Found {len(sample_names)} samples to evaluate")
    print("=" * 80)
    
    results = []
    
    for sample_name in sorted(sample_names):
        print(f"\nProcessing {sample_name}...")
        result = process_sample(sample_name, samples_dir, gt_dir, output_dir)
        
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        
        results.append(result)
        
        print(f"  Threshold:  IoU={result['threshold']['iou']:.4f}, Acc={result['threshold']['accuracy']:.4f}, F1={result['threshold']['f1']:.4f}")
        print(f"  GrabCut:    IoU={result['grabcut']['iou']:.4f}, Acc={result['grabcut']['accuracy']:.4f}, F1={result['grabcut']['f1']:.4f}")
        print(f"  Improvement: IoU={result['improvement']['iou_delta']:+.4f}, Acc={result['improvement']['accuracy_delta']:+.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        avg_thresh_iou = np.mean([r["threshold"]["iou"] for r in results])
        avg_gc_iou = np.mean([r["grabcut"]["iou"] for r in results])
        avg_thresh_acc = np.mean([r["threshold"]["accuracy"] for r in results])
        avg_gc_acc = np.mean([r["grabcut"]["accuracy"] for r in results])
        
        print(f"\nAverage Threshold IoU:   {avg_thresh_iou:.4f}")
        print(f"Average GrabCut IoU:     {avg_gc_iou:.4f}")
        print(f"Average Improvement IoU: {avg_gc_iou - avg_thresh_iou:+.4f}")
        print(f"\nAverage Threshold Acc:   {avg_thresh_acc:.4f}")
        print(f"Average GrabCut Acc:     {avg_gc_acc:.4f}")
        print(f"Average Improvement Acc: {avg_gc_acc - avg_thresh_acc:+.4f}")
        
        # Per-sample table
        print("\n" + "-" * 100)
        print(f"{'Sample':<25} {'Thresh IoU':>10} {'GC IoU':>10} {'Δ IoU':>8} {'Thresh Acc':>10} {'GC Acc':>10} {'Δ Acc':>8}")
        print("-" * 100)
        for r in results:
            print(f"{r['sample']:<25} {r['threshold']['iou']:>10.4f} {r['grabcut']['iou']:>10.4f} "
                  f"{r['improvement']['iou_delta']:>+8.4f} {r['threshold']['accuracy']:>10.4f} "
                  f"{r['grabcut']['accuracy']:>10.4f} {r['improvement']['accuracy_delta']:>+8.4f}")
        
        # Save detailed results
        import json
        results_path = report_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = []
            for r in results:
                serializable_results.append({
                    "sample": r["sample"],
                    "threshold": {k: float(v) for k, v in r["threshold"].items()},
                    "grabcut": {k: float(v) for k, v in r["grabcut"].items()},
                    "improvement": {k: float(v) for k, v in r["improvement"].items()}
                })
            json.dump(serializable_results, f, indent=2)
        print(f"\nDetailed results saved to: {results_path}")
        print(f"Comparison images saved to: {output_dir / 'comparisons'}")


if __name__ == "__main__":
    main()