"""
Tests for the AI segmentation module.

The network itself is mocked (fast, offline). One integration test runs the real model
and is skipped automatically when `rembg` is not installed.
"""
import numpy as np
import pytest

from src import ai_segment
from src.ai_segment import (
    ai_alpha_matte,
    ai_pipeline,
    alpha_to_binary_mask,
    alpha_to_bgr,
    AI_MODELS,
)


def _fake_model(calls=None):
    """Stand-in for _run_model: a soft square in the middle of whatever size it receives."""
    def fake(image_rgb, model_name):
        if calls is not None:
            calls.append((image_rgb.shape, model_name))
        h, w = image_rgb.shape[:2]
        a = np.zeros((h, w), np.uint8)
        a[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
        a[h // 4 - 1, w // 4: 3 * w // 4] = 128  # one soft edge row
        return a
    return fake


@pytest.fixture
def image():
    return np.random.RandomState(0).randint(0, 255, (80, 120, 3), dtype=np.uint8)


class TestAiAlphaMatte:
    def test_shape_dtype_matches_input(self, image, monkeypatch):
        monkeypatch.setattr(ai_segment, "_run_model", _fake_model())
        alpha = ai_alpha_matte(image)
        assert alpha.shape == image.shape[:2]
        assert alpha.dtype == np.uint8

    def test_large_image_downscaled_then_restored(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ai_segment, "_run_model", _fake_model(calls))
        big = np.zeros((400, 800, 3), np.uint8)
        alpha = ai_alpha_matte(big, max_dimension=200)
        assert max(calls[0][0][:2]) == 200            # model saw the small version
        assert alpha.shape == (400, 800)              # output is original size

    def test_small_image_not_resized(self, image, monkeypatch):
        calls = []
        monkeypatch.setattr(ai_segment, "_run_model", _fake_model(calls))
        ai_alpha_matte(image, max_dimension=1024)
        assert calls[0][0][:2] == image.shape[:2]

    @pytest.mark.parametrize("key", list(AI_MODELS))
    def test_model_key_is_translated(self, image, monkeypatch, key):
        calls = []
        monkeypatch.setattr(ai_segment, "_run_model", _fake_model(calls))
        ai_alpha_matte(image, model=key)
        assert calls[0][1] == AI_MODELS[key]

    def test_raw_model_name_passthrough(self, image, monkeypatch):
        calls = []
        monkeypatch.setattr(ai_segment, "_run_model", _fake_model(calls))
        ai_alpha_matte(image, model="some_other_model")
        assert calls[0][1] == "some_other_model"

    @pytest.mark.parametrize("bad", [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8), None])
    def test_rejects_non_bgr(self, bad):
        with pytest.raises(ValueError):
            ai_alpha_matte(bad)


class TestHelpers:
    def test_binary_mask_threshold(self):
        alpha = np.array([[0, 100, 127, 128, 255]], np.uint8)
        assert alpha_to_binary_mask(alpha).tolist() == [[0, 0, 0, 255, 255]]

    def test_binary_mask_values_only_0_255(self):
        alpha = np.random.RandomState(1).randint(0, 256, (20, 20), dtype=np.uint8)
        assert set(np.unique(alpha_to_binary_mask(alpha))) <= {0, 255}

    def test_alpha_to_bgr(self):
        vis = alpha_to_bgr(np.full((4, 5), 7, np.uint8))
        assert vis.shape == (4, 5, 3)
        assert (vis == 7).all()

    def test_pipeline_returns_alpha_and_mask(self, image, monkeypatch):
        monkeypatch.setattr(ai_segment, "_run_model", _fake_model())
        alpha, mask = ai_pipeline(image)
        assert alpha.shape == mask.shape == image.shape[:2]
        assert mask[40, 60] == 255 and mask[0, 0] == 0

    def test_is_available_returns_bool(self):
        assert isinstance(ai_segment.is_available(), bool)


class TestRealModel:
    """Runs the actual network. Skipped if rembg is missing."""

    def test_isolates_bright_object(self):
        pytest.importorskip("rembg")
        pytest.importorskip("onnxruntime")
        import cv2
        img = np.full((256, 256, 3), 40, np.uint8)
        cv2.circle(img, (128, 128), 70, (0, 200, 255), -1)
        alpha, mask = ai_pipeline(img, model="general")
        assert alpha.shape == (256, 256)
        assert mask[128, 128] == 255        # centre of the object kept
        assert mask[5, 5] == 0              # corner background removed
