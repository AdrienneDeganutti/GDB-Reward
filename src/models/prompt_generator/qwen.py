"""
Qwen prompt-generator backend.
"""
from __future__ import annotations

from ...utils.registry import PROMPT_GENERATORS
from .base import BasePromptGenerator

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


@PROMPT_GENERATORS.register("qwen")
class QwenPromptGenerator(BasePromptGenerator):
    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        config.setdefault("model_id", DEFAULT_MODEL_ID)
        super().__init__(config)