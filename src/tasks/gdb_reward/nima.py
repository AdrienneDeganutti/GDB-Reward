"""``nima`` -> NIMA aesthetic quality, human-aligned preference

No-reference *aesthetic* score: NIMA (Talebi & Milanfar, 2018) predicts the mean
opinion score (MOS) an image would receive on the AVA aesthetics dataset. Unlike
``image_reward`` (prompt-conditioned human preference) this measures pure visual
aesthetics independent of the prompt, so it adds an orthogonal quality signal to
the composite rather than a redundant one.

Implemented on top of ``pyiqa`` (IQA-PyTorch), which ships the pretrained NIMA
weights (InceptionResNetV2/AVA by default) and downloads them on first use. If
``pyiqa`` is unavailable ``load()`` raises and the metric is dropped from the
composite with a warning (like DreamSim / ImageReward).

Config keys (all optional; read from the shared reward/metric config dict):
  * ``device``       : torch device (shared with the other metrics; default cuda).
  * ``nima_variant`` : pyiqa metric name (default ``nima``; e.g. ``nima-vgg16-ava``).
  * ``nima_max_mos`` : MOS upper bound used to normalise the reward (default 10.0).
"""
from __future__ import annotations

from .base import BaseMetric


class NimaMetric(BaseMetric):
    """No-reference NIMA aesthetic MOS for the generated image (higher = better)."""

    key = "nima"
    needs_reference = False

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.device = self.config.get("device", "cuda")
        self.variant = self.config.get("nima_variant", "nima")
        # AVA MOS lives in [1, 10]; used to map the raw MOS into a [0, 1] reward.
        self.max_mos = float(self.config.get("nima_max_mos", 10.0))
        self.model = None

    def load(self):
        import pyiqa
        import torch
        if not torch.cuda.is_available():
            self.device = "cpu"
        # Downloads the pretrained NIMA weights on first use; raises if pyiqa or
        # the weights are unavailable so the composite drops the metric.
        self.model = pyiqa.create_metric(self.variant, device=self.device)
        return self

    def compute(self, record, image, prompt, reference=None):
        import torch
        from torchvision.transforms.functional import to_tensor
        tensor = to_tensor(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mos = self.model(tensor)
        # pyiqa returns a (1, 1) / (1,) tensor; reduce to a python float.
        return float(mos.flatten()[0].item())

    def to_reward(self, raw: float) -> float:
        # MOS in [1, max_mos] -> [0, 1]. Monotonic, so it preserves the ranking
        # GRPO actually uses; the exact scale is not critical.
        denom = max(self.max_mos - 1.0, 1e-6)
        return float(max(0.0, min(1.0, (raw - 1.0) / denom)))