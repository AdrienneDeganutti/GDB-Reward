"""``ssim`` -> SSIM, structural similarity (reference-based).

Mean structural similarity between the generated image and the ground-truth
render (higher = more similar), computed with scikit-image. Reference-based:
returns ``None`` when no render is available. Both images are resized to a
common size (longest side capped by ``ssim_max_side`` for speed) before scoring.
"""
from __future__ import annotations

from .base import BaseMetric


class Ssim(BaseMetric):
    """Structural similarity to the reference render (higher = closer)."""

    key = "ssim"
    needs_reference = True

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.max_side = int(self.config.get("ssim_max_side", 512))

    def load(self):
        # Fail loudly at load time (so the metric is dropped) if scikit-image
        # isn't installed, rather than per-sample.
        from skimage.metrics import structural_similarity  # noqa: F401
        return self

    def _prep(self, image, size):
        import numpy as np
        return np.asarray(image.convert("RGB").resize(size), dtype="float32")

    def compute(self, record, image, prompt, reference=None):
        if reference is None:
            return None
        from skimage.metrics import structural_similarity
        w, h = image.size
        scale = min(1.0, self.max_side / max(w, h))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        a = self._prep(image, size)
        b = self._prep(reference, size)
        return float(structural_similarity(a, b, channel_axis=2, data_range=255.0))

    def to_reward(self, raw: float) -> float:
        # SSIM is in [-1, 1] (typically [0, 1]); map linearly to [0, 1].
        return float(max(0.0, min(1.0, (raw + 1.0) / 2.0)))