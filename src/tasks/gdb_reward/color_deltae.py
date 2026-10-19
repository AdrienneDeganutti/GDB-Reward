"""``color_deltae`` -> Color distance, ΔE CIEDE2000 (paper Sec. 2.1 / App. A).

Mean CIEDE2000 distance from each ground-truth layout colour to the nearest
colour in the generated image's palette. Lower is better; the paper considers
ΔE < 5 perceptually acceptable.
"""
from __future__ import annotations

import math

import numpy as np

from .base import BaseMetric
from .color_utils import ciede2000, image_palette, layout_target_colors


class ColorDeltaE(BaseMetric):
    """Mean CIEDE2000 distance from each layout colour to the image palette.

    For every ground-truth layout colour we take the nearest colour in the
    generated image's palette and average those ΔE distances. Lower is better;
    the paper considers ΔE < 5 perceptually acceptable.
    """

    key = "color_deltae"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.k = int(self.config.get("palette_colors", 6))
        self.scale = float(self.config.get("deltae_scale", 20.0))

    def compute(self, record, image, prompt, reference=None):
        targets = layout_target_colors(record)
        if not targets:
            return None
        palette = [rgb for rgb, _ in image_palette(image, self.k)]
        if not palette:
            return None
        dists = [min(ciede2000(t, p) for p in palette) for t in targets]
        return float(np.mean(dists))

    def to_reward(self, raw: float) -> float:
        return float(math.exp(-raw / self.scale))