from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._classes: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        def deco(cls: type) -> type:
            key = name.lower()
            if key in self._classes:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._classes[key] = cls
            return cls
        return deco

    def get(self, name: str) -> type:
        key = str(name).lower()
        if key not in self._classes:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Registered: {sorted(self._classes)}")
        return self._classes[key]

    def available(self) -> list[str]:
        return sorted(self._classes)


# Global registries for each swappable component family.
PROMPT_GENERATORS: "Registry[Any]" = Registry("prompt_generator")
IMAGE_GENERATORS: "Registry[Any]" = Registry("image_generator")
REWARDS: "Registry[Any]" = Registry("reward")


def build_from_config(registry: "Registry[Any]", config: dict, kind: str | None = None) -> Any:
    if config is None or "name" not in config:
        raise ValueError(f"{kind or registry.kind} config must contain a 'name' key")
    cfg = {k: v for k, v in config.items() if k != "name"}
    cls = registry.get(config["name"])
    return cls(cfg)