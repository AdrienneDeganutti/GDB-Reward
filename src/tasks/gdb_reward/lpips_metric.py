"""``lpips`` -> LPIPS, learned perceptual image similarity (reference-based).

LPIPS distance between the generated image and the ground-truth render (lower =
more perceptually similar). Requires the ``lpips`` package; if it (or its small
backbone weights) is unavailable, ``load()`` raises and the metric is dropped
from the composite with a warning (like DreamSim / ImageReward).
"""
from __future__ import annotations

from .base import BaseMetric


class Lpips(BaseMetric):
    """Learned perceptual similarity to the reference render (lower = closer)."""

    key = "lpips"
    needs_reference = True

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.net = self.config.get("lpips_net", "alex")  # alex | vgg | squeeze
        self.device = self.config.get("device", "cuda")
        self.model = None

    def load(self):
        import lpips as _lpips
        import torch
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.model = _lpips.LPIPS(net=self.net).to(self.device).eval()
        return self

    def _to_tensor(self, image, size):
        import numpy as np
        import torch
        arr = np.asarray(image.convert("RGB").resize(size), dtype="float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW in [0,1]
        return (t * 2.0 - 1.0).to(self.device)                   # LPIPS wants [-1,1]

    def compute(self, record, image, prompt, reference=None):
        if reference is None:
            return None
        import torch
        size = image.size  # compare at the generated image's resolution
        a = self._to_tensor(image, size)
        b = self._to_tensor(reference, size)
        with torch.no_grad():
            d = self.model(a, b).item()
        return float(d)

    def to_reward(self, raw: float) -> float:
        return float(max(0.0, min(1.0, 1.0 - raw)))