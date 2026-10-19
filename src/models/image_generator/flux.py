"""Flux text-to-image backend (diffusers).

Wraps a diffusers Flux pipeline behind ``BaseImageGenerator``. Defaults to
``black-forest-labs/FLUX.1-dev``; set ``model_id`` to a FLUX.2 checkpoint and the
pipeline class is auto-selected (or force it with ``pipeline_class``).

"""
from __future__ import annotations

from typing import Optional, Sequence

from PIL.Image import Image

from ...utils.registry import IMAGE_GENERATORS
from .base import BaseImageGenerator

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"


def _silence_clip_truncation_warning() -> None:
    """Drop diffusers' benign "CLIP can only handle 77 tokens" truncation warning.

    FLUX encodes the prompt with two text encoders: CLIP (hard 77-token cap, used
    only for a single pooled global embedding) and T5 (up to ``max_sequence_length``
    = 512, which carries the full detailed prompt). Our layout prompts fit inside
    T5, so CLIP's truncation drops no real detail -- but diffusers logs it on every
    batched pipeline call, flooding the training logs. We filter out ONLY that
    message and deliberately keep the separate T5 ``max_sequence_length`` warning,
    which would flag genuine prompt-length loss.
    """
    import logging

    needle = "CLIP can only handle sequences up to"

    class _DropClipTruncation(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            return needle not in record.getMessage()

    try:
        # Force diffusers to configure its library root logger + default handler,
        # so the filter below actually has a handler to attach to.
        from diffusers.utils import logging as dlog
        dlog.get_verbosity()
    except Exception:
        pass

    # The warning is emitted by a child logger (diffusers.pipelines.flux.*) and
    # propagates to the "diffusers" root logger's handler, so filtering there
    # catches it regardless of the exact pipeline module (FLUX.1 vs FLUX.2).
    root = logging.getLogger("diffusers")
    for handler in root.handlers:
        if not any(isinstance(f, _DropClipTruncation) for f in handler.filters):
            handler.addFilter(_DropClipTruncation())


@IMAGE_GENERATORS.register("flux")
class FluxImageGenerator(BaseImageGenerator):
    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        config.setdefault("model_id", DEFAULT_MODEL_ID)
        config.setdefault("guidance_scale", 3.5)
        super().__init__(config)
        self.device_map = self.config.get("device_map")  # e.g. "balanced" to shard across GPUs
        self.enable_cpu_offload = bool(self.config.get("enable_model_cpu_offload", False))
        self.pipeline_class = self.config.get("pipeline_class")  # optional override
        self.max_sequence_length = int(self.config.get("max_sequence_length", 512))

    def _resolve_pipeline_class(self):
        import diffusers
        if self.pipeline_class:
            return getattr(diffusers, self.pipeline_class)
        mid = self.model_id.lower()
        if "flux.2" in mid or "flux2" in mid:
            return getattr(diffusers, "Flux2Pipeline")
        return getattr(diffusers, "FluxPipeline")

    def load(self) -> "FluxImageGenerator":
        from ...utils import logging as log
        log.info(f"Loading image generator '{self.model_id}' ...")
        pipe_cls = self._resolve_pipeline_class()

        from_kwargs = {"torch_dtype": self._torch_dtype()}
        if self.device_map:
            # Shard components across the visible GPUs at load time. When a
            # device_map is used the pipeline is already placed, so we must not
            # call .to() or enable_model_cpu_offload() afterwards.
            if self.enable_cpu_offload:
                log.warn("Flux: both device_map and enable_model_cpu_offload set; "
                         "using device_map and ignoring cpu offload.")
            from_kwargs["device_map"] = self.device_map

        self.pipe = pipe_cls.from_pretrained(self.model_id, **from_kwargs)

        if self.device_map:
            log.info(f"Flux: sharded across GPUs (device_map={self.device_map!r}).")
        elif self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        # CLIP's 77-token cap truncates our long prompts, but T5 (max_sequence_length)
        # carries the full detail, so the per-call warning is benign noise -- drop it.
        _silence_clip_truncation_warning()
        return self

    def _generator(self, seed: Optional[int]):
        """A seeded torch.Generator (or None). CPU-placed when the pipeline is
        sharded/offloaded, since diffusers seeds latents on CPU then moves them."""
        if seed is None:
            return None
        import torch
        device = self.device if not (self.device_map or self.enable_cpu_offload) else "cpu"
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
        # FLUX.1 supports a longer T5 sequence; ignore if the pipeline doesn't accept it.
        try:
            return self.pipe(max_sequence_length=self.max_sequence_length, **kwargs)
        except TypeError:
            return self.pipe(**kwargs)

    def generate(self, prompt: str, seed: Optional[int] = None) -> Image:
        if self.pipe is None:
            raise RuntimeError("FluxImageGenerator.load() must be called before generate().")
        return self._run(prompt, self._generator(seed)).images[0]

    def generate_batch(self, prompts: Sequence[str],
                       seeds: Optional[Sequence[Optional[int]]] = None) -> list[Image]:
        """Generate every prompt in a single batched denoising pass.

        Passing all G group prompts through one pipeline call runs the diffusion
        loop once over the batch instead of G times, which is markedly faster than
        looping over :meth:`generate` -- especially with ``device_map`` sharding,
        where it amortises the cross-GPU transfer overhead over the batch. A
        per-prompt generator list preserves independent seeding.
        """
        if self.pipe is None:
            raise RuntimeError("FluxImageGenerator.load() must be called before generate().")
        prompts = list(prompts)
        if not prompts:
            return []
        if seeds is None:
            seeds = [None] * len(prompts)
        gens = [self._generator(s) for s in seeds]
        # If no prompt is seeded, pass a bare None so diffusers samples freely.
        generator = gens if any(g is not None for g in gens) else None
        return list(self._run(prompts, generator).images)