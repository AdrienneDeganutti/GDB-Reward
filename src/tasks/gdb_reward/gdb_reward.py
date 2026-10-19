"""GDBReward: a composite, GraphicDesignBench-derived reward for the RL loop.

This is the reward that scores a generated image against the *source layout* 
using a configurable mix of metrics drawn from the GraphicDesignBench taxonomy 
(see ``metrics.py``), and folds them into a single scalar in ``[0, 1]`` that 
GRPO turns into a group-relative advantage.

Design notes
------------
* **No backprop through the eval.** Every metric runs in inference only and
  returns a plain float. GRPO needs nothing more than a scalar per sample, so
  the reward stack can be non-differentiable (CIEDE2000, OCR, a VLM judge, ...).
* **Composite & re-normalising.** The reward is ``sum_k w_k * m_k`` over the
  metrics that produced a value for *this* sample. If a metric is not applicable
  (e.g. a reference-based metric with no render) or fails to load (missing
  optional dependency), it is dropped and the remaining weights are renormalised
  so the reward stays in ``[0, 1]``.
* **Built for group-relative advantage.** Only the *ranking* of the G prompts
  for a given layout matters (GRPO standardises within the group), so prefer
  metrics with within-group variance. The zero-dependency colour/harmony
  metrics already vary across prompts, which makes the loop trainable out of the
  box with no model downloads.

A metric is included if its weight is non-zero AND it is not in
``disabled_metrics``. The remaining weights are renormalised per sample, so
turning one off simply redistributes its share across the rest. Prefer 
``disabled_metrics`` over zeroing a weight so the run's intent is explicit 
and the original weighting is preserved for comparison.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PIL import Image

from ...utils import logging as log
from ...utils.registry import REWARDS
from ...utils.reward import BaseReward
from .metrics import METRIC_CLASSES, BaseMetric

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...utils.data import LayoutRecord

# Sensible default if no weights are given: the zero-dependency colour metric that
# runs anywhere and produces within-group variance immediately (real runs set an
# explicit `weights` block in the config).
_DEFAULT_WEIGHTS = {"color_deltae": 1.0}
_REFERENCE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


@REWARDS.register("gdb")
class GDBReward(BaseReward):
    """Weighted combination of GraphicDesignBench metrics -> scalar reward."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        weights = self.config.get("weights") or _DEFAULT_WEIGHTS
        # Explicit on/off switch (for ablations): any metric listed under
        # `disabled_metrics` is dropped from the reward even if it still carries a
        # weight, so a metric can be toggled off without editing the weights block.
        # A typo here is a hard error, because a silently-ignored disable would
        # corrupt the ablation it was meant to run.
        disabled = {str(k) for k in (self.config.get("disabled_metrics") or [])}
        bad_disabled = disabled - set(METRIC_CLASSES)
        if bad_disabled:
            raise ValueError(
                f"GDBReward: disabled_metrics has unknown metric(s) "
                f"{sorted(bad_disabled)}. Available: {sorted(METRIC_CLASSES)}")
        self.weights: dict[str, float] = {
            k: float(v) for k, v in weights.items()
            if float(v) != 0.0 and k not in disabled}
        if not self.weights:
            raise ValueError("GDBReward: no metrics enabled (all weights 0).")
        unknown = set(self.weights) - set(METRIC_CLASSES)
        if unknown:
            raise ValueError(
                f"GDBReward: unknown metric(s) {sorted(unknown)}. "
                f"Available: {sorted(METRIC_CLASSES)}")
        if disabled:
            log.info(f"GDBReward: metrics disabled via config: {sorted(disabled)}")

        rd = self.config.get("renders_dir")
        self.renders_dir: Optional[Path] = Path(rd) if rd else None
        self.metrics: dict[str, BaseMetric] = {}
        self._ref_cache: dict[str, Optional[Image.Image]] = {}
        # layout_id -> render path, built once from renders_dir (see _build_ref_index)
        self._ref_index: Optional[dict[str, Path]] = None

    def load(self) -> "GDBReward":
        for key in list(self.weights):
            metric = METRIC_CLASSES[key](self.config)
            try:
                self.metrics[key] = metric.load()
            except Exception as exc:  # missing optional dep / download failure
                dropped = self.weights.pop(key)
                log.warn(f"GDBReward: metric '{key}' unavailable ({exc!r}); "
                         f"dropping it (weight {dropped}).")
        if not self.metrics:
            raise RuntimeError("GDBReward: no metrics could be loaded.")
        needs_ref = any(m.needs_reference for m in self.metrics.values())
        if needs_ref:
            self._ref_index = self._build_ref_index()
            if not self._ref_index:
                log.warn("GDBReward: reference-based metric enabled but no renders "
                         f"found under 'renders_dir' ({self.renders_dir}); those "
                         "metrics will be skipped per sample.")
            else:
                log.info(f"GDBReward: indexed {len(self._ref_index)} reference "
                         f"render(s) under {self.renders_dir}")
        log.info("GDBReward ready. Metrics: "
                 + ", ".join(f"{k}={self.weights[k]:g}" for k in self.metrics))
        return self

    def _build_ref_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if self.renders_dir is None or not self.renders_dir.is_dir():
            return index
        for p in sorted(self.renders_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _REFERENCE_EXTS:
                index.setdefault(p.stem, p)
        return index

    def _reference(self, layout_id: str) -> Optional[Image.Image]:
        if self._ref_index is None:
            return None
        if layout_id in self._ref_cache:
            return self._ref_cache[layout_id]
        path = self._ref_index.get(layout_id)
        img = Image.open(path).convert("RGB") if path is not None else None
        self._ref_cache[layout_id] = img
        return img

    def score(self, record: "LayoutRecord", image: "Image.Image",
              prompt: str) -> dict[str, Any]:
        need_ref = any(m.needs_reference for m in self.metrics.values())
        reference = self._reference(record.id) if need_ref else None

        contributions: list[tuple[float, float]] = []  # (weight, reward_in_[0,1])
        info: dict[str, Any] = {}
        for key, metric in self.metrics.items():
            try:
                raw = metric.compute(record, image, prompt, reference)
            except Exception as exc:
                log.warn(f"GDBReward: metric '{key}' failed on {record.id}: {exc!r}")
                raw = None
            if raw is None:
                info[key] = None
                continue
            r = float(metric.to_reward(raw))
            info[key] = round(raw, 4)
            info[f"{key}_r"] = round(r, 4)
            contributions.append((self.weights[key], r))

        if not contributions:
            return {"reward": 0.0, "metrics_used": 0, **info}

        wsum = sum(w for w, _ in contributions)
        reward = sum(w * r for w, r in contributions) / wsum
        return {"reward": float(reward), "metrics_used": len(contributions), **info}