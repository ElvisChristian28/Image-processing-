"""
Tests for threshold module
"""
import pytest
import numpy as np
import cv2
from src.threshold import (
    convert_to_grayscale,
    apply_gaussian_blur,
    compute_threshold_mask,
    threshold_pipeline,
)


class TestConvertToGrayscale:
    def test_bgr_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = convert_to_grayscale(img)
        assert gray.shape == (100, 100)
        assert gray.dtype == np.uint8

    def test_bgra_image(self):
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        gray = convert_to_grayscale(img)
        assert gray.shape == (100, 100)
        assert gray.dtype == np.uint8

    def test_invalid_channels_1(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            convert_to_grayscale(img)

    def test_invalid_channels_5(self):
        img = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            convert_to_grayscale(img)

    def test_non_uint8(self):
        img = np.random.rand(100, 100, 3).astype(np.float32)
        with pytest.raises(ValueError):
            convert_to_grayscale(img)


class TestApplyGaussianBlur:
    def test_basic(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        blur = apply_gaussian_blur(gray, ksize=5)
        assert blur.shape == gray.shape
        assert blur.dtype == np.uint8

    def test_invalid_ksize_even(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_gaussian_blur(gray, ksize=4)

    def test_invalid_ksize_small(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_gaussian_blur(gray, ksize=1)


class TestComputeThresholdMask:
    def test_otsu(self):
        # Create image with clear foreground/background
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[20:80, 20:80] = 255  # White square
        mask = compute_threshold_mask(gray, method="otsu")
        assert mask.shape == (100, 100)
        assert set(np.unique(mask)).issubset({0, 255})
        # Should detect the square
        assert np.sum(mask[20:80, 20:80] == 255) > 0

    def test_adaptive_mean(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mask = compute_threshold_mask(gray, method="adaptive_mean")
        assert mask.shape == (100, 100)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_adaptive_gaussian(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mask = compute_threshold_mask(gray, method="adaptive_gaussian")
        assert mask.shape == (100, 100)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_unknown_method(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_threshold_mask(gray, method="unknown")


class TestThresholdPipeline:
    def test_end_to_end_otsu(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = threshold_pipeline(img, method="otsu")
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_end_to_end_adaptive(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = threshold_pipeline(img, method="adaptive_mean")
        assert mask.shape == (100, 100)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_with_bgra(self):
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        mask = threshold_pipeline(img, method="otsu")
        assert mask.shape == (100, 100)