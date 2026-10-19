"""``dreamsim`` -> DreamSim, perceptual quality, reference-based

Human-aligned perceptual similarity to the reference render (lower = closer).
Reference-based: returns ``None`` for samples with no render. Requires the
``dreamsim`` package; if missing, ``load()`` raises and the metric is dropped
from the composite with a warning.
"""
from __future__ import annotations

import os

from .base import BaseMetric


class DreamSim(BaseMetric):
    """Human-aligned perceptual similarity to the reference render (lower=closer).

    Reference-based: returns ``None`` for samples with no render. Requires the
    ``dreamsim`` package; if missing, ``load()`` raises and the metric is
    dropped from the composite with a warning.
    """

    key = "dreamsim"
    needs_reference = True

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.variant = self.config.get("dreamsim_type", "ensemble")
        self.device = self.config.get("device", "cuda")
        # Cache weights in a stable location (dreamsim defaults to ``./models``
        # in the cwd, which would pollute wherever the process is launched).
        self.cache_dir = self.config.get(
            "dreamsim_cache_dir",
            os.path.expanduser("~/.cache/dreamsim"))
        self.model = None
        self.preprocess = None

    def load(self):
        import torch
        from dreamsim import dreamsim
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.model, self.preprocess = dreamsim(
            pretrained=True, device=self.device,
            dreamsim_type=self.variant, cache_dir=self.cache_dir)
        return self

    def compute(self, record, image, prompt, reference=None):
        if reference is None:
            return None
        import torch
        a = self.preprocess(image.convert("RGB")).to(self.device)
        b = self.preprocess(reference.convert("RGB")).to(self.device)
        with torch.no_grad():
            d = self.model(a, b).item()
        return float(d)

    def to_reward(self, raw: float) -> float:
        return float(max(0.0, min(1.0, 1.0 - raw)))