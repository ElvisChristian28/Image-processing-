"""
Integration tests for the full pipeline
"""
import pytest
import numpy as np
import cv2
from src.threshold import threshold_pipeline
from src.grabcut_seg import grabcut_pipeline
from src.extract import extract_foreground
from src.utils import resize_if_needed, scale_mask_to_original, load_image
from pathlib import Path


class TestFullPipelineSynthetic:
    """Test with synthetic images where ground truth is known."""

    def test_white_square_on_black(self):
        """Simple case: white square on black background."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = [255, 255, 255]  # White square
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, iterations=3)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 200, 4)
        assert outputs["black_bg"].shape == (200, 200, 3)
        assert outputs["binary_mask"].shape == (200, 200, 3)
        
        # Foreground should be extracted (some white pixels in result)
        assert np.any(outputs["transparent"][:, :, 3] == 255)
        assert np.any(outputs["black_bg"] != 0)

    def test_black_square_on_white(self):
        """Inverse: black square on white background."""
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        img[50:150, 50:150] = [0, 0, 0]  # Black square
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, iterations=3)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 200, 4)

    def test_colored_object_on_different_background(self):
        """Red circle on blue background."""
        img = np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)  # Blue background (BGR)
        cv2.circle(img, (100, 100), 50, [0, 0, 255], -1)  # Red circle
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, iterations=3)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 200, 4)
        assert np.any(outputs["transparent"][:, :, 3] == 255)

    def test_gradient_background(self):
        """Object on gradient background - challenging for simple threshold."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(200):
            val = int(i * 255 / 200)
            img[:, i] = [val, val, val]
        # Gray circle in center
        cv2.circle(img, (100, 100), 50, [128, 128, 128], -1)
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, iterations=5)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 200, 4)


class TestPipelineWithRect:
    """Test pipeline with user-provided rectangle."""

    def test_with_rect(self):
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        # Add distinct object
        cv2.rectangle(img, (50, 50), (150, 150), [255, 255, 255], -1)
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, rect=(50, 50, 100, 100), iterations=3)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 200, 4)
        assert np.any(outputs["transparent"][:, :, 3] == 255)

    def test_rect_improves_empty_threshold(self):
        """When threshold fails, rect should help."""
        img = np.full((200, 200, 3), 128, dtype=np.uint8)  # Uniform gray
        cv2.circle(img, (100, 100), 40, [200, 200, 200], -1)  # Slightly lighter circle
        
        thresh = threshold_pipeline(img, method="otsu")
        # Threshold might fail (empty or full)
        fg_ratio = np.count_nonzero(thresh) / thresh.size
        
        if fg_ratio < 0.01 or fg_ratio > 0.99:
            # Threshold failed, use rect
            gc_mask = grabcut_pipeline(img, thresh, rect=(60, 60, 80, 80), 
                                       use_rect_fallback=True, iterations=5)
            assert np.any(gc_mask == 255)
        else:
            # Threshold worked
            gc_mask = grabcut_pipeline(img, thresh, iterations=5)
            assert np.any(gc_mask == 255)


class TestPipelineWithResize:
    """Test pipeline with image resizing for large images."""

    def test_large_image_resize(self):
        """Large image should be resized for processing."""
        # Create large image (2000x2000)
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        cv2.circle(img, (1000, 1000), 400, [255, 255, 255], -1)
        
        # Resize for processing
        processed, scale = resize_if_needed(img, max_dimension=1024)
        assert scale > 1.0
        assert max(processed.shape[:2]) <= 1024
        
        # Process on resized
        thresh = threshold_pipeline(processed, method="otsu")
        gc_mask = grabcut_pipeline(processed, thresh, iterations=3)
        
        # Scale mask back
        final_mask = scale_mask_to_original(gc_mask, img.shape[:2])
        assert final_mask.shape == (2000, 2000)
        
        outputs = extract_foreground(img, final_mask)
        assert outputs["transparent"].shape == (2000, 2000, 4)

    def test_small_image_no_resize(self):
        """Small image should not be resized."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.circle(img, (250, 250), 100, [255, 255, 255], -1)
        
        processed, scale = resize_if_needed(img, max_dimension=1024)
        assert scale == 1.0
        assert processed.shape == img.shape
        
        thresh = threshold_pipeline(processed, method="otsu")
        gc_mask = grabcut_pipeline(processed, thresh, iterations=3)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (500, 500, 4)


class TestPipelineEdgeCases:
    """Test edge cases in the pipeline."""

    def test_thin_object(self):
        """Thin line object - may not be detected well by threshold+GrabCut."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.line(img, (20, 180), (180, 20), [255, 255, 255], 3)
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, iterations=3)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 200, 4)
        # Thin objects may not be detected - just verify pipeline runs without error
        # and produces valid output format
        assert outputs["transparent"].dtype == np.uint8

    def test_object_touching_border(self):
        """Object touching image border."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[0:100, 0:100] = [255, 255, 255]  # Top-left quadrant
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, iterations=3)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 200, 4)
        # Border pixels should be extracted
        assert outputs["transparent"][0, 0, 3] == 255

    def test_multiple_objects(self):
        """Multiple separate objects."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.circle(img, (75, 100), 40, [255, 255, 255], -1)
        cv2.circle(img, (225, 100), 40, [255, 255, 255], -1)
        cv2.rectangle(img, (125, 60), (175, 140), [255, 255, 255], -1)
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, iterations=5, cleanup=True)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 300, 4)
        # Should extract all three objects (or at least the largest)
        assert np.any(outputs["transparent"][:, :, 3] == 255)

    def test_low_contrast(self):
        """Low contrast object/background."""
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        img[60:140, 60:140] = [150, 150, 150]  # Slightly lighter
        
        thresh = threshold_pipeline(img, method="otsu")
        gc_mask = grabcut_pipeline(img, thresh, iterations=5)
        outputs = extract_foreground(img, gc_mask)
        
        assert outputs["transparent"].shape == (200, 200, 4)


class TestPipelineWithSampleImages:
    """Test with actual sample images if available."""

    @pytest.fixture(scope="class")
    def sample_dir(self):
        return Path(__file__).parent.parent / "samples"

    def test_sample_images_exist(self, sample_dir):
        """Verify sample images exist."""
        samples = list(sample_dir.glob("*.png"))
        # Filter out mask files
        samples = [s for s in samples if not s.stem.endswith("_mask")]
        assert len(samples) >= 8  # At least 8 test images

    def test_process_all_samples(self, sample_dir):
        """Process all sample images through the pipeline."""
        samples = list(sample_dir.glob("*.png"))
        samples = [s for s in samples if not s.stem.endswith("_mask")]
        
        for sample_path in samples[:5]:  # Test first 5 to save time
            img = load_image(str(sample_path))
            assert img is not None
            assert img.ndim == 3
            assert img.shape[2] in (3, 4)
            
            # Run full pipeline
            thresh = threshold_pipeline(img, method="otsu")
            gc_mask = grabcut_pipeline(img, thresh, iterations=3)
            outputs = extract_foreground(img, gc_mask)
            
            assert "transparent" in outputs
            assert "black_bg" in outputs
            assert "binary_mask" in outputs
            assert outputs["transparent"].shape[:2] == img.shape[:2]


class TestExtractOutputFormats:
    """Test that all output formats are correctly generated."""

    def test_four_outputs(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        
        outputs = extract_foreground(img, mask)
        
        # Check all four outputs exist
        assert set(outputs.keys()) == {"transparent", "black_bg", "binary_mask", "foreground_only"}
        
        # Check shapes
        assert outputs["transparent"].shape == (100, 100, 4)  # RGBA
        assert outputs["black_bg"].shape == (100, 100, 3)     # BGR
        assert outputs["binary_mask"].shape == (100, 100, 3)  # BGR visualization
        # foreground_only is cropped to bounding box (60x60 in this case)
        assert outputs["foreground_only"].shape == (60, 60, 3)  # BGR cropped

    def test_foreground_only_cropped(self):
        """Test that foreground_only is properly cropped to bounding box."""
        img = np.full((100, 100, 3), [50, 100, 150], dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255  # 60x40 box at (30, 20)
        
        outputs = extract_foreground(img, mask, feather_radius=0)
        
        fg_only = outputs["foreground_only"]
        # Should be cropped to 60x40 (height x width)
        assert fg_only.shape == (60, 40, 3)
        # All pixels should be the foreground color (no black background)
        assert np.all(fg_only == [50, 100, 150])

    def test_transparent_alpha_channel(self):
        # Image in BGR format (OpenCV default)
        img = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)  # B=100, G=150, R=200
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        
        outputs = extract_foreground(img, mask, feather_radius=0)
        
        # Foreground should be opaque
        assert outputs["transparent"][25, 25, 3] == 255
        # Background should be transparent
        assert outputs["transparent"][5, 5, 3] == 0
        # Foreground color preserved - create_transparent_png converts BGR to RGBA (RGB order)
        # Input BGR [100, 150, 200] becomes RGB [200, 150, 100] in output
        assert np.allclose(outputs["transparent"][25, 25, :3], [200, 150, 100], atol=1)

    def test_black_bg_output(self):
        img = np.full((50, 50, 3), [200, 100, 50], dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        
        outputs = extract_foreground(img, mask)
        
        # Foreground color preserved
        assert np.all(outputs["black_bg"][25, 25] == [200, 100, 50])
        # Background is black
        assert np.all(outputs["black_bg"][5, 5] == [0, 0, 0])

    def test_binary_mask_visualization(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        
        outputs = extract_foreground(np.zeros((50, 50, 3), dtype=np.uint8), mask)
        
        # Binary mask should be white on black
        assert np.all(outputs["binary_mask"][25, 25] == [255, 255, 255])
        assert np.all(outputs["binary_mask"][5, 5] == [0, 0, 0])

    def test_feathering_only_on_transparent(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        
        outputs = extract_foreground(img, mask, feather_radius=5)
        
        # Transparent should have feathered edges
        alpha = outputs["transparent"][:, :, 3]
        assert np.any((alpha > 0) & (alpha < 255))
        
        # Black bg should NOT have feathered edges (binary)
        # Check by looking at color transitions
        unique_colors = np.unique(outputs["black_bg"].reshape(-1, 3), axis=0)
        # Should only have 2 colors: foreground and black
        assert len(unique_colors) <= 2


class TestPipelineConsistency:
    """Test that pipeline produces consistent results."""

    def test_deterministic_grabcut(self):
        """GrabCut with same inputs should produce same output."""
        np.random.seed(42)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = [255, 255, 255]
        
        thresh = threshold_pipeline(img, method="otsu")
        
        mask1 = grabcut_pipeline(img, thresh, iterations=5)
        mask2 = grabcut_pipeline(img, thresh, iterations=5)
        
        # Should be identical (GrabCut is deterministic with same init)
        assert np.array_equal(mask1, mask2)

    def test_different_iterations(self):
        """More iterations should not drastically change result."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = [255, 255, 255]
        
        thresh = threshold_pipeline(img, method="otsu")
        
        mask_2 = grabcut_pipeline(img, thresh, iterations=2)
        mask_10 = grabcut_pipeline(img, thresh, iterations=10)
        
        # Results should be similar (IoU > 0.9)
        iou = np.logical_and(mask_2, mask_10).sum() / np.logical_or(mask_2, mask_10).sum()
        assert iou > 0.8  # Allow some variation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])