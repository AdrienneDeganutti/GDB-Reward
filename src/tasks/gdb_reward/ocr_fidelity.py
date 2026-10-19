"""
Whether the generated image *reads back* as the layout's intended text. We run
an off-the-shelf OCR system (EasyOCR: detection + recognition in one pass) over
the generated image, then, for every ground-truth text string in the layout, take
the best normalised similarity to what OCR recovered. The metric is the mean of
those per-string scores, so it is recall-oriented over the intended strings
(extra text OCR happens to find is not penalised), matching the paper's framing
of "is the intended target string recognised".

Notes on the wider *text-fidelity* suite: Font-Family / Text-Align accuracy,
Font-Size MAE and Letter-Spacing MAE all require the paper's custom
*Text-Params-Predictor* (a model that recovers typographic parameters from
rendered text patches), which is not shipped in this repo -- so they are not
implemented here. Colour distance (ΔE CIEDE2000) is already covered by the
``color_deltae`` metric.

Only needs ``easyocr`` (weights download on first use); no reference render.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .base import BaseMetric

# EasyOCR readers are heavy (own detector + recogniser); share one per config.
_READER_CACHE: dict[tuple, object] = {}


def _get_reader(langs: tuple[str, ...], gpu: bool):
    cache_key = (langs, gpu)
    reader = _READER_CACHE.get(cache_key)
    if reader is None:
        import easyocr
        reader = easyocr.Reader(list(langs), gpu=gpu)
        _READER_CACHE[cache_key] = reader
    return reader


def _norm(s: str) -> str:
    """Lower-case and collapse whitespace so OCR spacing quirks don't hurt."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _match_score(target: str, candidates: list[str], full_text: str) -> float:
    """Best in [0,1] of (a) pairwise ratio to any recognised string and
    (b) how much of ``target`` appears as a contiguous block in the full OCR
    text. (b) gives credit when OCR splits/merges lines differently."""
    pairwise = max((SequenceMatcher(None, target, c).ratio() for c in candidates),
                   default=0.0)
    containment = 0.0
    if full_text:
        block = SequenceMatcher(None, target, full_text).find_longest_match(
            0, len(target), 0, len(full_text))
        containment = block.size / max(1, len(target))
    return max(pairwise, containment)


class OcrTextFidelity(BaseMetric):
    """Mean OCR read-back similarity of the layout's ground-truth text strings.

    For each intended text string we take the best similarity against what an
    off-the-shelf OCR system recovers from the generated image, then average.
    With ``ocr_match_threshold`` > 0 the per-string scores are binarised first,
    turning the metric into the paper's exact "OCR Accuracy" *rate*; the default
    (0) returns the softer mean similarity, which is friendlier as a reward.
    """

    key = "ocr_accuracy"
    needs_reference = False

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        langs = self.config.get("ocr_langs", ["en"])
        self.langs = tuple(langs) if isinstance(langs, (list, tuple)) else (langs,)
        self.device = self.config.get("device", "cuda")
        self.gpu = bool(self.config.get("ocr_gpu", str(self.device).startswith("cuda")))
        self.min_conf = float(self.config.get("ocr_min_conf", 0.3))
        self.match_threshold = float(self.config.get("ocr_match_threshold", 0.0))
        self.reader = None

    def load(self):
        try:
            import torch
            if not torch.cuda.is_available():
                self.gpu = False
        except Exception:
            self.gpu = False
        self.reader = _get_reader(self.langs, self.gpu)
        return self

    def _ocr_strings(self, image) -> list[str]:
        import numpy as np
        arr = np.array(image.convert("RGB"))
        out = []
        for _, text, conf in self.reader.readtext(arr):
            if conf >= self.min_conf and text.strip():
                out.append(text)
        return out

    def compute(self, record, image, prompt, reference=None):
        targets = [_norm(t) for t in record.text_strings()]
        targets = [t for t in targets if t]
        if not targets:
            return None  # image-only design: OCR fidelity is N/A

        recognized = [_norm(s) for s in self._ocr_strings(image)]
        recognized = [r for r in recognized if r]
        full_text = " ".join(recognized)

        scores = [_match_score(t, recognized, full_text) for t in targets]
        if self.match_threshold > 0.0:
            scores = [1.0 if s >= self.match_threshold else 0.0 for s in scores]
        return float(sum(scores) / len(scores))

    def to_reward(self, raw: float) -> float:
        return float(max(0.0, min(1.0, raw)))