"""Text-to-image generators (prompt -> image)."""
from ...utils.registry import IMAGE_GENERATORS, build_from_config
from .base import BaseImageGenerator
from .flux import FluxImageGenerator  # noqa: F401  (registers "flux")
from .sdxl import SDXLImageGenerator  # noqa: F401  (registers "sdxl")


def build_image_generator(config: dict) -> BaseImageGenerator:
    """Instantiate the image generator named in ``config`` (does not call load())."""
    return build_from_config(IMAGE_GENERATORS, config, kind="image_generator")


__all__ = ["BaseImageGenerator", "FluxImageGenerator", "SDXLImageGenerator", "build_image_generator"]