"""
Tests for utils module
"""
import pytest
import numpy as np
import cv2
import tempfile
import os
from src.utils import (
    load_image,
    validate_image,
    resize_if_needed,
    scale_mask_to_original,
    create_comparison_grid,
    image_to_bytes,
    bytes_to_image,
    ensure_bgr,
)


class TestLoadImage:
    def test_load_existing_png(self):
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = np.full((100, 100, 3), 128, dtype=np.uint8)
            cv2.imwrite(f.name, img)
            temp_path = f.name
        
        try:
            loaded = load_image(temp_path)
            assert loaded.shape == (100, 100, 3)
            assert loaded.dtype == np.uint8
            assert np.all(loaded == 128)
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")

    def test_load_corrupted(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b"not an image")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError):
                load_image(temp_path)
        finally:
            os.unlink(temp_path)


class TestValidateImage:
    def test_valid_bgr(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        h, w, c = validate_image(img)
        assert (h, w, c) == (100, 100, 3)

    def test_valid_bgra(self):
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        h, w, c = validate_image(img)
        assert (h, w, c) == (100, 100, 4)

    def test_invalid_1d(self):
        img = np.random.randint(0, 255, (100,), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image(img)

    def test_invalid_2d(self):
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image(img)

    def test_invalid_4d(self):
        img = np.random.randint(0, 255, (1, 100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image(img)

    def test_invalid_channels_1(self):
        img = np.random.randint(0, 255, (100, 100, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image(img)

    def test_invalid_channels_5(self):
        img = np.random.randint(0, 255, (100, 100, 5), dtype=np.uint8)
        with pytest.raises(ValueError):
            validate_image(img)

    def test_invalid_dtype_float(self):
        img = np.random.rand(100, 100, 3).astype(np.float32)
        with pytest.raises(ValueError):
            validate_image(img)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            validate_image("not an array")


class TestResizeIfNeeded:
    def test_no_resize_needed(self):
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        resized, scale = resize_if_needed(img, max_dimension=1024)
        assert scale == 1.0
        assert resized.shape == img.shape
        assert np.array_equal(resized, img)

    def test_downscale_width(self):
        img = np.random.randint(0, 255, (500, 2000, 3), dtype=np.uint8)
        resized, scale = resize_if_needed(img, max_dimension=1024)
        assert scale > 1.0
        assert max(resized.shape[:2]) <= 1024
        # Aspect ratio preserved
        expected_h = int(500 * 1024 / 2000)
        assert resized.shape[0] == expected_h
        assert resized.shape[1] == 1024

    def test_downscale_height(self):
        img = np.random.randint(0, 255, (2000, 500, 3), dtype=np.uint8)
        resized, scale = resize_if_needed(img, max_dimension=1024)
        assert scale > 1.0
        assert max(resized.shape[:2]) <= 1024
        expected_w = int(500 * 1024 / 2000)
        assert resized.shape[1] == expected_w
        assert resized.shape[0] == 1024

    def test_downscale_both(self):
        img = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        resized, scale = resize_if_needed(img, max_dimension=1024)
        assert scale > 1.0
        assert max(resized.shape[:2]) == 1024
        # Width is larger, so width should be 1024
        assert resized.shape[1] == 1024
        assert resized.shape[0] == int(2000 * 1024 / 3000)

    def test_custom_interpolation(self):
        img = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        resized, scale = resize_if_needed(img, max_dimension=100, interpolation=cv2.INTER_NEAREST)
        assert resized.shape == (100, 100, 3)

    def test_bgra_image(self):
        img = np.random.randint(0, 255, (2000, 2000, 4), dtype=np.uint8)
        resized, scale = resize_if_needed(img, max_dimension=100)
        assert resized.shape == (100, 100, 4)
        assert scale > 1.0


class TestScaleMaskToOriginal:
    def test_no_scaling_needed(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        scaled = scale_mask_to_original(mask, (100, 100))
        assert scaled.shape == (100, 100)
        assert np.array_equal(scaled, mask)

    def test_upscale(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        scaled = scale_mask_to_original(mask, (100, 100))
        assert scaled.shape == (100, 100)
        # Should be binary after thresholding
        assert set(np.unique(scaled)).issubset({0, 255})
        # Region should be roughly doubled
        assert np.sum(scaled == 255) > 0

    def test_downscale(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        scaled = scale_mask_to_original(mask, (50, 50))
        assert scaled.shape == (50, 50)
        assert set(np.unique(scaled)).issubset({0, 255})

    def test_nearest_interpolation(self):
        # Binary mask should use NEAREST to preserve sharp edges
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        scaled = scale_mask_to_original(mask, (100, 100), interpolation=cv2.INTER_NEAREST)
        assert scaled.shape == (100, 100)
        # With NEAREST, values should be exactly 0 or 255
        assert set(np.unique(scaled)).issubset({0, 255})

    def test_non_binary_input(self):
        # Input with intermediate values should be thresholded
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 128  # Not 255
        scaled = scale_mask_to_original(mask, (100, 100))
        assert set(np.unique(scaled)).issubset({0, 255})


class TestCreateComparisonGrid:
    def test_basic_grid(self):
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
        titles = ["A", "B", "C", "D"]
        
        grid = create_comparison_grid(images, titles, cols=4)
        
        assert grid.shape[2] == 3  # BGR
        assert grid.dtype == np.uint8
        # Height should match input (approximately)
        assert grid.shape[0] > 0

    def test_multiple_rows(self):
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(6)]
        titles = [f"Img{i}" for i in range(6)]
        
        grid = create_comparison_grid(images, titles, cols=4)
        
        assert grid.shape[2] == 3
        # Should have 2 rows

    def test_rgba_images(self):
        images = [np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8) for _ in range(2)]
        titles = ["A", "B"]
        
        grid = create_comparison_grid(images, titles, cols=2)
        
        assert grid.shape[2] == 3  # Output is always BGR

    def test_single_image(self):
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
        titles = ["Single"]
        
        grid = create_comparison_grid(images, titles, cols=1)
        
        assert grid.shape[2] == 3


class TestImageToBytes:
    def test_png_encoding(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        data = image_to_bytes(img, ".png")
        assert isinstance(data, bytes)
        assert len(data) > 0
        # Verify it's valid PNG
        assert data[:8] == b'\x89PNG\r\n\x1a\n'

    def test_jpg_encoding(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        data = image_to_bytes(img, ".jpg")
        assert isinstance(data, bytes)
        assert len(data) > 0
        # Verify it's valid JPEG
        assert data[:2] == b'\xff\xd8'

    def test_rgba_png(self):
        img = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
        data = image_to_bytes(img, ".png")
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_round_trip(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        data = image_to_bytes(img, ".png")
        decoded = bytes_to_image(data)
        assert decoded.shape == img.shape
        assert decoded.dtype == np.uint8


class TestBytesToImage:
    def test_decode_png(self):
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        data = cv2.imencode('.png', img)[1].tobytes()
        decoded = bytes_to_image(data)
        assert decoded.shape == (50, 50, 3)
        assert np.all(decoded == 100)

    def test_decode_grayscale(self):
        img = np.full((50, 50), 150, dtype=np.uint8)
        data = cv2.imencode('.png', img)[1].tobytes()
        decoded = bytes_to_image(data)
        # Should be converted to BGR
        assert decoded.shape == (50, 50, 3)
        assert np.all(decoded == 150)

    def test_decode_invalid(self):
        with pytest.raises(ValueError):
            bytes_to_image(b"not an image")


class TestEnsureBgr:
    def test_already_bgr(self):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = ensure_bgr(img)
        assert np.array_equal(result, img)

    def test_bgra_to_bgr(self):
        img = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)
        result = ensure_bgr(img)
        assert result.shape == (50, 50, 3)

    def test_grayscale_to_bgr(self):
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = ensure_bgr(img)
        assert result.shape == (50, 50, 3)
        # All channels should be equal (grayscale)
        assert np.all(result[:, :, 0] == result[:, :, 1])
        assert np.all(result[:, :, 1] == result[:, :, 2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])