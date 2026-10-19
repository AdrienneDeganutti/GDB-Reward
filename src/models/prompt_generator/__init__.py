"""Prompt-generator policies (layout metadata -> image prompt)."""
from ...utils.registry import PROMPT_GENERATORS, build_from_config
from .base import BasePromptGenerator
from .qwen import QwenPromptGenerator  # noqa: F401  (registers "qwen")


def build_prompt_generator(config: dict) -> BasePromptGenerator:
    """Instantiate the prompt generator named in ``config`` and load its weights."""
    gen = build_from_config(PROMPT_GENERATORS, config, kind="prompt_generator")
    return gen


__all__ = ["BasePromptGenerator", "QwenPromptGenerator", "build_prompt_generator"]