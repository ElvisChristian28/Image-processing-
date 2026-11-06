"""
AI segmentation module: remove the background with a local neural network (rembg / ONNX).

Runs fully offline after the first model download (~170 MB, cached in ~/.u2net).
No API key, account or GPU required.

`rembg` is imported lazily, so the classical Threshold + GrabCut pipeline keeps working
even when `rembg` is not installed. Use `is_available()` to check.
"""
from functools import lru_cache
from typing import Dict

import cv2
import numpy as np

# Friendly name -> rembg model name
AI_MODELS: Dict[str, str] = {
    "people": "u2net_human_seg",       # best for portraits / people
    "general": "isnet-general-use",    # objects, products, animals
}
DEFAULT_MODEL = "people"


def is_available() -> bool:
    """True if the optional `rembg` dependency can be imported."""
    try:
        import rembg  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=4)
def _get_session(model_name: str):
    """Create (and cache) a rembg session so the model is only loaded once."""
    from rembg import new_session
    return new_session(model_name)


def _run_model(image_rgb: np.ndarray, model_name: str) -> np.ndarray:
    """
    Run the network and return a soft alpha matte, shape (H, W), uint8 0-255.

    This is the only function that touches rembg, which keeps it easy to mock in tests.
    """
    from PIL import Image
    from rembg import remove

    session = _get_session(model_name)
    result = remove(Image.fromarray(image_rgb), session=session, only_mask=True)
    return np.array(result.convert("L"), dtype=np.uint8)


def ai_alpha_matte(
    image: np.ndarray,
    model: str = DEFAULT_MODEL,
    max_dimension: int = 1024,
) -> np.ndarray:
    """
    Predict a soft alpha matte (0 = background, 255 = foreground) for a BGR image.

    The image is downscaled for inference if its longest side exceeds `max_dimension`,
    and the matte is scaled back (bilinear) to the original resolution, so the result
    always has the same height/width as the input.

    Args:
        image: BGR image, shape (H, W, 3), uint8
        model: Key of AI_MODELS ("people" / "general") or a raw rembg model name
        max_dimension: Longest side used for inference (limits time/memory)

    Returns:
        Alpha matte, shape (H, W), uint8
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("ai_alpha_matte expects a 3-channel BGR image")

    model_name = AI_MODELS.get(model, model)
    h, w = image.shape[:2]

    work = image
    longest = max(h, w)
    if longest > max_dimension:
        s = max_dimension / longest
        work = cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))),
                          interpolation=cv2.INTER_AREA)

    alpha = _run_model(cv2.cvtColor(work, cv2.COLOR_BGR2RGB), model_name)

    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return alpha.astype(np.uint8)


def alpha_to_binary_mask(alpha: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Convert a soft alpha matte to a hard {0, 255} mask (used for stats / mask panel)."""
    return np.where(alpha > threshold, 255, 0).astype(np.uint8)


def alpha_to_bgr(alpha: np.ndarray) -> np.ndarray:
    """Grayscale visualisation of the matte as a 3-channel BGR image."""
    return cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)


def ai_pipeline(
    image: np.ndarray,
    model: str = DEFAULT_MODEL,
    max_dimension: int = 1024,
):
    """
    Full AI step. Returns (alpha, mask): soft matte and hard binary mask, both (H, W) uint8.
    """
    alpha = ai_alpha_matte(image, model=model, max_dimension=max_dimension)
    return alpha, alpha_to_binary_mask(alpha)
