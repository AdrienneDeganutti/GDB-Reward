"""Base class for the (frozen) text-to-image generator.

The image generator turns the policy's natural-language prompt into an image. It
is never trained in this pipeline, so the interface is deliberately tiny: load
once, then ``generate(prompt) -> PIL.Image``. Any backend (Flux, SDXL, an API)
implements this and is selected via ``image_generator.name`` in the config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from PIL.Image import Image


class BaseImageGenerator(ABC):
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model_id: str = self.config.get("model_id", "")
        self.dtype_str: str = self.config.get("dtype", "bfloat16")
        self.device: str = self.config.get("device", "cuda")
        self.height: int = int(self.config.get("height", 1024))
        self.width: int = int(self.config.get("width", 1024))
        self.num_inference_steps: int = int(self.config.get("num_inference_steps", 20))
        self.guidance_scale: float = float(self.config.get("guidance_scale", 3.5))
        self.pipe = None

    def _torch_dtype(self):
        import torch
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}.get(self.dtype_str, torch.bfloat16)

    @abstractmethod
    def load(self) -> "BaseImageGenerator":
        ...

    @abstractmethod
    def generate(self, prompt: str, seed: Optional[int] = None) -> Image:
        ...

    def generate_batch(self, prompts: Sequence[str],
                       seeds: Optional[Sequence[Optional[int]]] = None) -> list[Image]:
        """Generate one image per prompt.

        Default implementation just loops over :meth:`generate`; backends that can
        run a whole batch through a single forward pass (e.g. Flux) should override
        this for a large speedup, especially for GRPO's per-layout group.
        """
        if seeds is None:
            seeds = [None] * len(prompts)
        return [self.generate(p, seed=s) for p, s in zip(prompts, seeds)]

    def close(self) -> None:
        """Free GPU memory held by the pipeline."""
        import gc
        self.pipe = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass