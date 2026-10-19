"""The small interface every GDB metric implements.

Every metric is a :class:`BaseMetric`:

  * ``compute(record, image, prompt, reference)`` returns the metric's value in
    its *natural* units (e.g. a CIEDE2000 distance, an ImageReward score), or ``None``
    when the metric is not applicable to this sample (e.g. a reference-based
    metric with no reference render). Higher or lower may be better.
  * ``to_reward(raw)`` maps that natural value to ``[0, 1]`` where **higher is
    always better**, so the composite reward can sum metrics directly.

This split keeps the raw, paper-comparable number around for logging while the
trainer only ever sees the normalised contribution. Metrics with heavy/optional
dependencies (DreamSim, ImageReward) import them lazily in ``load()`` so
importing a metric module stays cheap and never fails when an extra isn't
installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...utils.data import LayoutRecord

RGB = tuple[int, int, int]


class BaseMetric(ABC):
    """A single scalar measurement of a generated image against a layout."""

    key: str = "metric"
    needs_reference: bool = False

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def load(self) -> "BaseMetric":
        """Load any models. Default: no-op. May raise to signal it's unavailable."""
        return self

    @abstractmethod
    def compute(self, record: "LayoutRecord", image: Image.Image, prompt: str,
                reference: Optional[Image.Image] = None) -> Optional[float]:
        """Return the raw metric value (natural units), or None if N/A."""

    @abstractmethod
    def to_reward(self, raw: float) -> float:
        """Map a raw value to a reward in [0, 1] (higher is better)."""