from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .registry import REWARDS, build_from_config
from . import logging as log

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image
    from .data import LayoutRecord


class BaseReward(ABC):
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def load(self) -> "BaseReward":
        """Hook for loading any models (captioner, metric nets). Default: no-op."""
        return self

    @abstractmethod
    def score(self, record: "LayoutRecord", image: "Image", prompt: str) -> dict[str, Any]:
        ...

    def close(self) -> None:
        """Free any held resources. Default: no-op."""


@REWARDS.register("placeholder")
class PlaceholderReward(BaseReward):
    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.mode = (self.config or {}).get("mode", "random")
        self.target_words = int((self.config or {}).get("target_words", 60))
        log.warn(
            "Using PlaceholderReward (mode=%s): this is NOT a real eval signal. "
            "Set reward.name=gdb to use the real GraphicDesignBench composite "
            "(src/tasks/gdb_reward)." % self.mode)

    def score(self, record: "LayoutRecord", image: "Image", prompt: str) -> dict[str, Any]:
        if self.mode == "zero":
            reward = 0.0
        elif self.mode == "length":
            n = len(prompt.split())
            reward = max(0.0, 1.0 - abs(n - self.target_words) / self.target_words)
        else:  # random
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            reward = int(digest[:8], 16) / 0xFFFFFFFF
        return {"reward": float(reward), "placeholder": True, "mode": self.mode}


def _ensure_rewards_registered() -> None:
    try:
        import src.tasks.gdb_reward  # noqa: F401  (registers "gdb")
    except Exception as exc:  # pragma: no cover - import-time only
        log.warn(f"Could not import src.tasks.gdb_reward rewards: {exc!r}")


def build_reward(config: dict) -> BaseReward:
    """Instantiate the reward named in ``config`` (``{name: ..., ...}``)."""
    _ensure_rewards_registered()
    return build_from_config(REWARDS, config, kind="reward")