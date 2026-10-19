"""Stable Diffusion XL text-to-image backend (diffusers).

Wraps diffusers' ``StableDiffusionXLPipeline`` behind ``BaseImageGenerator`` so
SDXL is a drop-in image-generator backend for the GRPO pipeline, parallel to
``flux.py``. Select it with ``image_generator.name: sdxl``.

"""
from __future__ import annotations

from typing import Optional, Sequence

from PIL.Image import Image

from ...utils.registry import IMAGE_GENERATORS
from .base import BaseImageGenerator

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def _silence_clip_truncation_warning() -> None:
    """Drop diffusers' per-call "CLIP can only handle 77 tokens" warning.

    Unlike Flux (where CLIP truncation is benign because T5 carries the full
    prompt), for SDXL this truncation is real: both encoders are CLIP, so any
    prompt past ~77 tokens loses detail. We flag that once at load time (see
    :meth:`SDXLImageGenerator.load`) and suppress only the repeated per-call log
    line here to keep the training logs readable.
    """
    import logging

    needle = "CLIP can only handle sequences up to"

    class _DropClipTruncation(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            return needle not in record.getMessage()

    try:
        from diffusers.utils import logging as dlog
        dlog.get_verbosity()
    except Exception:
        pass

    root = logging.getLogger("diffusers")
    for handler in root.handlers:
        if not any(isinstance(f, _DropClipTruncation) for f in handler.filters):
            handler.addFilter(_DropClipTruncation())


@IMAGE_GENERATORS.register("sdxl")
class SDXLImageGenerator(BaseImageGenerator):
    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        config.setdefault("model_id", DEFAULT_MODEL_ID)
        # SDXL's recommended classifier-free guidance is ~5.0 (Flux uses 3.5); a
        # Flux-tuned 3.5 washes SDXL out. Steps default higher than Flux's 20.
        config.setdefault("guidance_scale", 5.0)
        config.setdefault("num_inference_steps", 30)
        super().__init__(config)
        self.enable_cpu_offload = bool(self.config.get("enable_model_cpu_offload", False))
        # Weight variant to download ("fp16" pulls the half-precision safetensors;
        # None uses the fp32 files). fp16 files load fine cast to our bf16 dtype.
        self.variant = self.config.get("variant", "fp16")
        self.pipeline_class = self.config.get("pipeline_class")  # optional override
        self.negative_prompt = self.config.get("negative_prompt")  # usually None for parity with Flux

    def _resolve_pipeline_class(self):
        import diffusers
        name = self.pipeline_class or "StableDiffusionXLPipeline"
        return getattr(diffusers, name)

    def load(self) -> "SDXLImageGenerator":
        from ...utils import logging as log
        log.info(f"Loading image generator '{self.model_id}' (SDXL) ...")
        pipe_cls = self._resolve_pipeline_class()

        from_kwargs = {"torch_dtype": self._torch_dtype(), "use_safetensors": True}
        if self.variant:
            from_kwargs["variant"] = self.variant
        try:
            self.pipe = pipe_cls.from_pretrained(self.model_id, **from_kwargs)
        except Exception as e:  # noqa: BLE001 -- variant may be unavailable for a given checkpoint
            if self.variant:
                log.warn(f"SDXL: variant={self.variant!r} unavailable ({e}); "
                         "retrying without a variant.")
                from_kwargs.pop("variant")
                self.pipe = pipe_cls.from_pretrained(self.model_id, **from_kwargs)
            else:
                raise

        if self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        log.warn("SDXL encodes prompts with CLIP (77-token cap per encoder) and has "
                 "no long-context text encoder; prompts longer than ~77 tokens are "
                 "truncated. Per-call truncation warnings are suppressed below.")
        _silence_clip_truncation_warning()
        return self

    def _generator(self, seed: Optional[int]):
        """A seeded ``torch.Generator`` (or None). CPU-placed when the pipeline is
        offloaded, since diffusers seeds latents on CPU before moving them."""
        if seed is None:
            return None
        import torch
        device = "cpu" if self.enable_cpu_offload else self.device
        return torch.Generator(device=device).manual_seed(int(seed))

    def _run(self, prompt, generator):
        """Call the pipeline with our fixed image settings for one prompt or a
        list of prompts (diffusers returns one image per prompt)."""
        kwargs = dict(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        if self.negative_prompt is not None:
            n = len(prompt) if isinstance(prompt, (list, tuple)) else 1
            kwargs["negative_prompt"] = (
                [self.negative_prompt] * n if n > 1 else self.negative_prompt
            )
        return self.pipe(**kwargs)

    def generate(self, prompt: str, seed: Optional[int] = None) -> Image:
        if self.pipe is None:
            raise RuntimeError("SDXLImageGenerator.load() must be called before generate().")
        return self._run(prompt, self._generator(seed)).images[0]

    def generate_batch(self, prompts: Sequence[str],
                       seeds: Optional[Sequence[Optional[int]]] = None) -> list[Image]:
        """Generate every prompt in a single batched denoising pass.

        Passing all G group prompts through one pipeline call runs the diffusion
        loop once over the batch instead of G times, matching the Flux backend's
        fast path -- important for GRPO throughput. A per-prompt generator list
        preserves independent seeding.
        """
        if self.pipe is None:
            raise RuntimeError("SDXLImageGenerator.load() must be called before generate().")
        prompts = list(prompts)
        if not prompts:
            return []
        if seeds is None:
            seeds = [None] * len(prompts)
        gens = [self._generator(s) for s in seeds]
        generator = gens if any(g is not None for g in gens) else None
        return list(self._run(prompts, generator).images)