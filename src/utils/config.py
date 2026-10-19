from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class Config(dict):
    """A dict that also supports attribute access and recursive wrapping.

    ``cfg.grpo.steps`` and ``cfg["grpo"]["steps"]`` are equivalent. Missing keys
    raise ``AttributeError`` (for attribute access) so typos fail loudly.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return _wrap(value)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return _wrap(super().get(key, default))

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(dict(out[key]), dict(value))
        else:
            out[key] = value
    return out


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML rules (so 10, 1e-5, true, [a,b] work)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data)}")
    return data


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a config file, resolve an optional ``model_config`` include, apply overrides.

    Args:
        path: Path to the (training or inference) config YAML.
        overrides: Iterable of ``dotted.key=value`` strings applied last.

    The included model config is merged so its keys (``prompt_generator``,
    ``image_generator``, ``reward``, ...) sit at the top level next to the
    training keys. Keys already present in the training config win, so a training
    config can override individual model fields inline.
    """
    path = Path(path)
    data = load_yaml(path)

    model_config_ref = data.pop("model_config", None)
    if model_config_ref is not None:
        model_path = (path.parent / model_config_ref).resolve()
        model_data = load_yaml(model_path)
        # Training-config values override the included model config.
        data = _deep_merge(model_data, data)
        data["_model_config_path"] = str(model_path)

    data["_config_path"] = str(path.resolve())

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key.subkey=value")
        key, raw_value = item.split("=", 1)
        _set_dotted(data, key.strip(), _coerce_scalar(raw_value.strip()))

    return _wrap(data)