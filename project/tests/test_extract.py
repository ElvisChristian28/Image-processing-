"""
Tests for extract module
"""
import pytest
import numpy as np
import cv2
from src.extract import (
    apply_mask_to_image,
    create_transparent_png,
    create_binary_mask_image,
    extract_foreground,
    save_outputs,
    feather_edges,
)


class TestFeatherEdges:
    def test_basic_feathering(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        feathered = feather_edges(mask, radius=5)

        assert feathered.shape == (100, 100)
        assert feathered.dtype == np.uint8
        # Center should be fully opaque
        assert feathered[50, 50] == 255
        # Outside should be fully transparent
        assert feathered[10, 10] == 0
        # Edge should have intermediate values
        assert 0 < feathered[22, 22] < 255
        assert 0 < feathered[78, 78] < 255

    def test_disabled_feathering(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        feathered = feather_edges(mask, radius=0)
        assert np.array_equal(feathered, mask)

        feathered = feather_edges(mask, radius=1)
        assert np.array_equal(feathered, mask)

    def test_even_radius_adjusted(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        # Even radius should be adjusted to odd
        feathered_4 = feather_edges(mask, radius=4)
        feathered_5 = feather_edges(mask, radius=5)
        assert np.array_equal(feathered_4, feathered_5)

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        feathered = feather_edges(mask, radius=5)
        assert np.all(feathered == 0)

    def test_full_mask(self):
        mask = np.full((50, 50), 255, dtype=np.uint8)
        feathered = feather_edges(mask, radius=5)
        assert np.all(feathered == 255)

    def test_thin_structure(self):
        # Very thin line - feathering should not crash
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, 10:40] = 255  # Horizontal line

        feathered = feather_edges(mask, radius=3)
        assert feathered.shape == (50, 50)
        # Line should still exist but feathered (1-pixel wide has no full-opacity center)
        assert np.max(feathered) > 0
        assert np.max(feathered) <= 255


class TestApplyMaskToImage:
    def test_basic(self):
        image = np.full((50, 50, 3), 255, dtype=np.uint8)  # White image
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        result = apply_mask_to_image(image, mask, background_color=(0, 0, 0))

        assert result.shape == (50, 50, 3)
        # Foreground should be white
        assert np.all(result[25, 25] == [255, 255, 255])
        # Background should be black
        assert np.all(result[5, 5] == [0, 0, 0])

    def test_custom_background_color(self):
        image = np.full((50, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        # Function uses BGR order for background_color
        result = apply_mask_to_image(image, mask, background_color=(0, 0, 255))  # Red in BGR

        assert np.all(result[25, 25] == [100, 100, 100])  # Foreground
        assert np.all(result[5, 5] == [0, 0, 255])  # Background (BGR)

    def test_mask_already_binary(self):
        image = np.full((20, 20, 3), 200, dtype=np.uint8)
        mask = np.full((20, 20), 1, dtype=np.uint8)  # 0/1 instead of 0/255

        result = apply_mask_to_image(image, mask)
        assert np.all(result == 200)


class TestCreateTransparentPng:
    def test_basic(self):
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        rgba = create_transparent_png(image, mask, feather_radius=0)

        assert rgba.shape == (50, 50, 4)
        assert rgba.dtype == np.uint8
        # Foreground opaque
        assert rgba[25, 25, 3] == 255
        assert np.all(rgba[25, 25, :3] == [128, 128, 128])
        # Background transparent
        assert rgba[5, 5, 3] == 0

    def test_with_feathering(self):
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        rgba = create_transparent_png(image, mask, feather_radius=5)

        # Edge should have partial transparency
        assert 0 < rgba[12, 12, 3] < 255
        assert 0 < rgba[38, 38, 3] < 255
        # Center fully opaque
        assert rgba[25, 25, 3] == 255
        # Outside fully transparent
        assert rgba[5, 5, 3] == 0

    def test_feathering_disabled(self):
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        rgba = create_transparent_png(image, mask, feather_radius=1)
        # Should be binary alpha
        assert set(np.unique(rgba[:, :, 3])).issubset({0, 255})


class TestCreateBinaryMaskImage:
    def test_basic(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        vis = create_binary_mask_image(mask)

        assert vis.shape == (50, 50, 3)
        assert vis.dtype == np.uint8
        # Should be 3-channel grayscale
        assert np.all(vis[25, 25] == [255, 255, 255])
        assert np.all(vis[5, 5] == [0, 0, 0])


class TestExtractForeground:
    def test_basic(self):
        image = np.full((50, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        outputs = extract_foreground(image, mask, feather_radius=0)

        assert "transparent" in outputs
        assert "black_bg" in outputs
        assert "binary_mask" in outputs

        assert outputs["transparent"].shape == (50, 50, 4)
        assert outputs["black_bg"].shape == (50, 50, 3)
        assert outputs["binary_mask"].shape == (50, 50, 3)

    def test_with_feathering(self):
        image = np.full((50, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        outputs = extract_foreground(image, mask, feather_radius=5)

        # Transparent should have feathered edges
        alpha = outputs["transparent"][:, :, 3]
        assert np.max(alpha) == 255
        assert np.min(alpha) == 0
        # Should have intermediate values
        intermediate = alpha[(alpha > 0) & (alpha < 255)]
        assert len(intermediate) > 0

    def test_black_bg_hard_edges(self):
        image = np.full((50, 50, 3), 200, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        outputs = extract_foreground(image, mask, feather_radius=5)

        # Black background should still have hard edges (binary mask)
        black_bg = outputs["black_bg"]
        # Check it's binary (no intermediate values from feathering)
        unique_colors = np.unique(black_bg.reshape(-1, 3), axis=0)
        assert len(unique_colors) <= 2  # Foreground color + background color


class TestSaveOutputs:
    def test_save_all_formats(self, tmp_path):
        image = np.full((20, 20, 3), 100, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:15] = 255

        outputs = extract_foreground(image, mask, feather_radius=3)
        saved = save_outputs(outputs, str(tmp_path), prefix="test_")

        assert "transparent" in saved
        assert "black_bg" in saved
        assert "binary_mask" in saved
        assert "foreground_only" in saved

        # Check files exist
        for path in saved.values():
            import os
            assert os.path.exists(path)

        # Transparent should be PNG
        assert saved["transparent"].endswith(".png")

    def test_save_creates_directory(self, tmp_path):
        image = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.full((10, 10), 255, dtype=np.uint8)

        outputs = extract_foreground(image, mask)
        new_dir = tmp_path / "new_subdir"
        saved = save_outputs(outputs, str(new_dir))

        import os
        assert os.path.exists(new_dir)
        assert len(saved) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestCropToForeground:
    """crop_to_foreground: tight crop + transparent background."""

    def _img_mask(self):
        from_img = np.full((100, 100, 3), 200, np.uint8)       # light "background"
        mask = np.zeros((100, 100), np.uint8)
        cv2.circle(mask, (50, 50), 20, 255, -1)
        return from_img, mask

    def test_is_cropped_bgra(self):
        from src.extract import crop_to_foreground
        img, mask = self._img_mask()
        out = crop_to_foreground(img, mask)
        assert out.ndim == 3 and out.shape[2] == 4
        assert out.shape[0] < 100 and out.shape[1] < 100       # cropped

    def test_background_pixels_fully_transparent(self):
        from src.extract import crop_to_foreground
        img, mask = self._img_mask()
        out = crop_to_foreground(img, mask)
        assert out[0, 0, 3] == 0                                # corner of the box = background
        assert (out[out[:, :, 3] == 0][:, :3] == 0).all()       # no colour leaks
        assert out[out.shape[0] // 2, out.shape[1] // 2, 3] == 255

    def test_empty_mask(self):
        from src.extract import crop_to_foreground
        out = crop_to_foreground(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint8))
        assert out.shape == (1, 1, 4) and out[0, 0, 3] == 0

    def test_padding_clamped_to_image(self):
        from src.extract import crop_to_foreground
        img, mask = self._img_mask()
        out = crop_to_foreground(img, mask, padding=500)
        assert out.shape[:2] == (100, 100)

    def test_soft_alpha_used_when_given(self):
        from src.extract import crop_to_foreground
        img, mask = self._img_mask()
        alpha = np.zeros((100, 100), np.uint8)
        alpha[mask > 0] = 128
        out = crop_to_foreground(img, mask, alpha=alpha)
        assert out[out.shape[0] // 2, out.shape[1] // 2, 3] == 128

    def test_alpha_shape_mismatch(self):
        from src.extract import crop_to_foreground
        img, mask = self._img_mask()
        with pytest.raises(ValueError):
            crop_to_foreground(img, mask, alpha=np.zeros((5, 5), np.uint8))


class TestCompositeAndAlphaPng:
    def test_composite_on_white(self):
        from src.extract import composite_on_background
        bgra = np.zeros((2, 2, 4), np.uint8)
        bgra[0, 0] = (10, 20, 30, 255)        # opaque stays
        out = composite_on_background(bgra, (255, 255, 255))
        assert out.shape == (2, 2, 3)
        assert tuple(out[0, 0]) == (10, 20, 30)
        assert tuple(out[1, 1]) == (255, 255, 255)   # transparent -> white

    def test_transparent_from_alpha(self):
        from src.extract import create_transparent_from_alpha
        img = np.full((6, 6, 3), 9, np.uint8)
        alpha = np.arange(36, dtype=np.uint8).reshape(6, 6)
        rgba = create_transparent_from_alpha(img, alpha)
        assert rgba.shape == (6, 6, 4)
        assert (rgba[:, :, 3] == alpha).all()

    def test_transparent_from_alpha_mismatch(self):
        from src.extract import create_transparent_from_alpha
        with pytest.raises(ValueError):
            create_transparent_from_alpha(np.zeros((4, 4, 3), np.uint8), np.zeros((3, 3), np.uint8))
