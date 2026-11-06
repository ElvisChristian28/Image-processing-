"""
Tests for grabcut_seg module
"""
import pytest
import numpy as np
import cv2
from src.grabcut_seg import (
    create_grabcut_mask_from_threshold,
    create_grabcut_mask_from_rect,
    run_grabcut,
    grabcut_to_binary_mask,
    grabcut_pipeline,
    morphological_cleanup,
    keep_largest_component,
    fill_holes,
    cleanup_mask,
)


class TestMorphologicalCleanup:
    def test_basic_open_close(self):
        # Create mask with noise and small hole
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # Main square
        mask[10, 10] = 255  # Noise pixel
        mask[45:48, 45:48] = 0  # Small hole in foreground (3x3)

        cleaned = morphological_cleanup(mask, open_kernel=3, close_kernel=5)

        # Noise should be removed
        assert cleaned[10, 10] == 0
        # Small hole should be filled
        assert cleaned[46, 46] == 255
        # Main region preserved
        assert cleaned[50, 50] == 255

    def test_open_only(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[5, 5] = 255  # Noise

        cleaned = morphological_cleanup(mask, open_kernel=3, close_kernel=0)
        assert cleaned[5, 5] == 0  # Noise removed
        assert cleaned[25, 25] == 255  # Main preserved

    def test_close_only(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[22:25, 22:25] = 0  # Small hole (3x3)

        cleaned = morphological_cleanup(mask, open_kernel=0, close_kernel=5)
        assert cleaned[23, 23] == 255  # Hole filled

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        cleaned = morphological_cleanup(mask)
        assert np.all(cleaned == 0)

    def test_full_mask(self):
        mask = np.full((50, 50), 255, dtype=np.uint8)
        cleaned = morphological_cleanup(mask)
        assert np.all(cleaned == 255)

    def test_invalid_open_kernel(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # Non-empty mask
        with pytest.raises(ValueError):
            morphological_cleanup(mask, open_kernel=2)  # Even

    def test_invalid_close_kernel(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # Non-empty mask
        with pytest.raises(ValueError):
            morphological_cleanup(mask, close_kernel=2)  # Even


class TestKeepLargestComponent:
    def test_multiple_components(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # Small component (400 px)
        mask[50:90, 50:90] = 255  # Large component (1600 px)

        cleaned = keep_largest_component(mask)

        # Small component removed
        assert np.all(cleaned[10:30, 10:30] == 0)
        # Large component kept
        assert np.all(cleaned[50:90, 50:90] == 255)

    def test_single_component(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        cleaned = keep_largest_component(mask)
        assert np.array_equal(cleaned, mask)

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        cleaned = keep_largest_component(mask)
        assert np.all(cleaned == 0)

    def test_full_mask(self):
        mask = np.full((50, 50), 255, dtype=np.uint8)
        cleaned = keep_largest_component(mask)
        assert np.all(cleaned == 255)

    def test_equal_components(self):
        # Two equal-sized components - should keep one (first found)
        mask = np.zeros((100, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # Component 1
        mask[60:90, 10:40] = 255  # Component 2 (same size)

        cleaned = keep_largest_component(mask)
        # Should have only one component
        assert np.count_nonzero(cleaned) == 900


class TestFillHoles:
    def test_simple_hole(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[20:30, 20:30] = 0  # Hole in center

        filled = fill_holes(mask)
        assert filled[25, 25] == 255  # Hole filled
        assert filled[15, 15] == 255  # Original foreground preserved
        assert filled[5, 5] == 0  # Background preserved

    def test_no_holes(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        filled = fill_holes(mask)
        assert np.array_equal(filled, mask)

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        filled = fill_holes(mask)
        assert np.all(filled == 0)

    def test_full_mask(self):
        mask = np.full((50, 50), 255, dtype=np.uint8)
        filled = fill_holes(mask)
        assert np.all(filled == 255)

    def test_border_foreground(self):
        # Foreground touching border - hole should still fill
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[0:30, 0:30] = 255  # Touching top-left border
        mask[10:20, 10:20] = 0  # Hole

        filled = fill_holes(mask)
        assert filled[15, 15] == 255  # Hole filled


class TestCleanupMask:
    def test_full_pipeline(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # Main object
        mask[5, 5] = 255  # Noise
        mask[45:48, 45:48] = 0  # Small hole

        cleaned = cleanup_mask(mask)

        # Noise removed
        assert cleaned[5, 5] == 0
        # Hole filled
        assert cleaned[46, 46] == 255
        # Main object preserved
        assert cleaned[50, 50] == 255

    def test_skip_steps(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[5, 5] = 255  # Noise

        # Disable open, keep largest should remove noise
        cleaned = cleanup_mask(mask, open_kernel=0, keep_largest=True)
        assert cleaned[5, 5] == 0

    def test_disable_keep_largest(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        mask[50:90, 50:90] = 255

        # Disable all morphological ops to preserve exact pixel counts
        cleaned = cleanup_mask(mask, keep_largest=False, open_kernel=0, close_kernel=0, fill_holes_flag=False)
        # Both components should remain
        assert np.count_nonzero(cleaned) == 400 + 1600

    def test_disable_fill_holes(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        mask[20:30, 20:30] = 0  # Hole

        cleaned = cleanup_mask(mask, fill_holes_flag=False, open_kernel=0, close_kernel=0)
        assert cleaned[25, 25] == 0  # Hole not filled


class TestCreateGrabcutMaskFromThreshold:
    def test_basic(self):
        thresh = np.zeros((50, 50), dtype=np.uint8)
        thresh[10:40, 10:40] = 255

        gc_mask = create_grabcut_mask_from_threshold(thresh)

        assert gc_mask.shape == (50, 50)
        assert gc_mask.dtype == np.uint8
        # Check values are valid GrabCut constants
        unique = np.unique(gc_mask)
        valid = {cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD}
        assert set(unique).issubset(valid)

    def test_definite_regions(self):
        thresh = np.zeros((50, 50), dtype=np.uint8)
        thresh[15:35, 15:35] = 255

        gc_mask = create_grabcut_mask_from_threshold(thresh, erode_kernel=3)

        # Center should be definite foreground (eroded by 1 pixel each side)
        # Original: 15:35, eroded: 16:34
        assert gc_mask[25, 25] == cv2.GC_FGD
        # Just inside original but outside eroded -> probable foreground
        assert gc_mask[15, 15] == cv2.GC_PR_FGD
        # Background should have definite background
        assert gc_mask[5, 5] == cv2.GC_BGD


class TestCreateGrabcutMaskFromRect:
    def test_basic(self):
        gc_mask = create_grabcut_mask_from_rect((100, 100), (20, 20, 40, 40))

        assert gc_mask.shape == (100, 100)
        # Inside rect -> probable foreground (shrink_factor=0.1 by default)
        # Rect: x=20, y=20, w=40, h=40 -> covers 20:60, 20:60
        # Shrunken: 24:56, 24:56 (approx)
        assert gc_mask[40, 40] in (cv2.GC_PR_FGD, cv2.GC_FGD)
        # Outside rect -> probable background
        assert gc_mask[10, 10] == cv2.GC_PR_BGD
        # Shrunken rect -> definite foreground
        assert gc_mask[30, 30] == cv2.GC_FGD

    def test_clamping(self):
        # Rect extending beyond image bounds
        gc_mask = create_grabcut_mask_from_rect((50, 50), (-10, -10, 80, 80))
        assert gc_mask.shape == (50, 50)
        # Should not crash and should have valid values


class TestRunGrabcut:
    def test_basic(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        init_mask = np.full((100, 100), cv2.GC_PR_BGD, dtype=np.uint8)
        init_mask[20:80, 20:80] = cv2.GC_PR_FGD
        init_mask[40:60, 40:60] = cv2.GC_FGD

        final_mask, bgd_model, fgd_model = run_grabcut(image, init_mask, iterations=2)

        assert final_mask.shape == (100, 100)
        assert final_mask.dtype == np.uint8
        assert bgd_model.shape == (1, 65)
        assert fgd_model.shape == (1, 65)


class TestGrabcutToBinaryMask:
    def test_conversion(self):
        gc_mask = np.full((50, 50), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[10:40, 10:40] = cv2.GC_PR_FGD
        gc_mask[20:30, 20:30] = cv2.GC_FGD
        gc_mask[5, 5] = cv2.GC_BGD

        binary = grabcut_to_binary_mask(gc_mask)

        assert binary.shape == (50, 50)
        assert set(np.unique(binary)).issubset({0, 255})
        # Foreground regions should be 255
        assert binary[25, 25] == 255
        assert binary[15, 15] == 255
        # Background should be 0
        assert binary[5, 5] == 0


class TestGrabcutPipeline:
    def test_with_threshold_mask(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Create distinct foreground/background
        image[20:80, 20:80] = [255, 255, 255]  # White square
        image[0:20, :] = [0, 0, 0]
        image[80:, :] = [0, 0, 0]
        image[:, 0:20] = [0, 0, 0]
        image[:, 80:] = [0, 0, 0]

        thresh = np.zeros((100, 100), dtype=np.uint8)
        thresh[20:80, 20:80] = 255

        mask = grabcut_pipeline(image, thresh, iterations=2)
        assert mask.shape == (100, 100)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_empty_threshold_fallback_rect(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        thresh = np.zeros((100, 100), dtype=np.uint8)  # Empty

        mask = grabcut_pipeline(image, thresh, rect=(20, 20, 40, 40), iterations=2)
        assert mask.shape == (100, 100)

    def test_cleanup_enabled(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image[20:80, 20:80] = [255, 255, 255]

        thresh = np.zeros((100, 100), dtype=np.uint8)
        thresh[20:80, 20:80] = 255
        thresh[5, 5] = 255  # Noise

        mask = grabcut_pipeline(image, thresh, iterations=2, cleanup=True)
        # Noise should be cleaned up
        assert mask[5, 5] == 0

    def test_cleanup_disabled(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image[20:80, 20:80] = [255, 255, 255]

        thresh = np.zeros((100, 100), dtype=np.uint8)
        thresh[20:80, 20:80] = 255
        thresh[5, 5] = 255  # Noise

        mask = grabcut_pipeline(image, thresh, iterations=2, cleanup=False)
        # Without cleanup, noise might remain (but GrabCut might also remove it)