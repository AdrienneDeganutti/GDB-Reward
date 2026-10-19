"""``image_reward`` -> ImageReward, human-aligned preference

No-reference human-preference score (ImageReward) for the ``(prompt, image)``
pair. Requires the ``image-reward`` package; dropped with a warning if missing.
"""
from __future__ import annotations

import math

from .base import BaseMetric


class ImageRewardMetric(BaseMetric):
    """No-reference human-preference score (ImageReward) for the (prompt, image).

    Requires the ``image-reward`` package; dropped with a warning if missing.
    """

    key = "image_reward"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.model_id = self.config.get("image_reward_id", "ImageReward-v1.0")
        self.device = self.config.get("device", "cuda")
        self.model = None

    @staticmethod
    def _patch_transformers_compat() -> None:
        """Back-fill symbols ImageReward's (2022-era) BLIP imports.

        ImageReward's bundled BLIP does ``from transformers.modeling_utils
        import apply_chunking_to_forward, ...`` -- those moved to
        ``transformers.pytorch_utils`` in transformers 4.0, and
        ``find_pruneable_heads_and_indices`` was removed outright in newer
        releases (>=5.x). Re-expose the moved ones and synthesize the removed
        one so the import works on modern transformers. Only fills what's missing.
        """
        import transformers.modeling_utils as mu
        try:
            import transformers.pytorch_utils as pu
        except Exception:
            pu = None
        for name in ("apply_chunking_to_forward", "prune_linear_layer",
                     "find_pruneable_heads_and_indices"):
            if not hasattr(mu, name) and pu is not None and hasattr(pu, name):
                setattr(mu, name, getattr(pu, name))

        if not hasattr(mu, "find_pruneable_heads_and_indices"):
            import torch

            def find_pruneable_heads_and_indices(heads, n_heads, head_size,
                                                 already_pruned_heads):
                """Canonical transformers helper (removed in transformers >=5)."""
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in heads:
                    head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                    mask[head] = 0
                mask = mask.view(-1).contiguous().eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index

            mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
            if pu is not None and not hasattr(pu, "find_pruneable_heads_and_indices"):
                pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

        # BLIP's init_tokenizer reads ``tokenizer.additional_special_tokens_ids``
        # (removed from the tokenizer base class in transformers >=5). Rebuild it
        # from the still-present ``all_special_tokens`` minus the standard ones.
        try:
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _B
        except Exception:
            _B = None
        if _B is not None and not hasattr(_B, "additional_special_tokens_ids"):
            def _additional_special_tokens(self):
                standard = set(filter(None, [
                    getattr(self, a, None) for a in (
                        "bos_token", "eos_token", "unk_token", "sep_token",
                        "pad_token", "cls_token", "mask_token")]))
                return [t for t in self.all_special_tokens if t not in standard]

            def _additional_special_tokens_ids(self):
                return [self.convert_tokens_to_ids(t)
                        for t in _additional_special_tokens(self)]

            if not hasattr(_B, "additional_special_tokens"):
                _B.additional_special_tokens = property(_additional_special_tokens)
            _B.additional_special_tokens_ids = property(_additional_special_tokens_ids)

        # transformers >=5 computes ``all_tied_weights_keys`` in ``post_init()``,
        # then ``tie_weights(recompute_mapping=False)`` reads it. BLIP's custom
        # BertModel ends ``__init__`` with the legacy ``self.init_weights()``
        # (which calls ``tie_weights``) *without* going through ``post_init``, so
        # the attribute is missing -> AttributeError. Back-fill it lazily on any
        # model that reaches ``tie_weights`` without it.
        if not getattr(mu.PreTrainedModel, "_gdb_all_tied_patch", False):
            _orig_tie = mu.PreTrainedModel.tie_weights

            def _tie_weights_safe(self, *args, **kwargs):
                if not hasattr(self, "all_tied_weights_keys"):
                    keys: dict = {}
                    getter = getattr(self, "get_expanded_tied_weights_keys", None)
                    if getter is not None:
                        try:
                            keys = getter(all_submodels=True)
                        except Exception:
                            keys = {}
                    self.all_tied_weights_keys = keys
                return _orig_tie(self, *args, **kwargs)

            mu.PreTrainedModel.tie_weights = _tie_weights_safe
            mu.PreTrainedModel._gdb_all_tied_patch = True

        # ``get_head_mask`` was dropped from ``ModuleUtilsMixin`` in transformers
        # >=5, but BLIP's BertModel.forward still calls ``self.get_head_mask(...)``
        # (``get_extended_attention_mask`` it overrides itself; ``invert_attention_mask``
        # is still provided). Re-add the canonical implementation if it's gone.
        mixin = getattr(mu, "ModuleUtilsMixin", None)
        if mixin is not None and not hasattr(mixin, "get_head_mask"):
            def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
                if head_mask.dim() == 1:
                    head_mask = (head_mask.unsqueeze(0).unsqueeze(0)
                                 .unsqueeze(-1).unsqueeze(-1))
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                assert head_mask.dim() == 5, (
                    f"head_mask.dim != 5, instead {head_mask.dim()}")
                return head_mask.to(dtype=self.dtype)

            def get_head_mask(self, head_mask, num_hidden_layers,
                              is_attention_chunked=False):
                if head_mask is not None:
                    head_mask = self._convert_head_mask_to_5d(
                        head_mask, num_hidden_layers)
                    if is_attention_chunked is True:
                        head_mask = head_mask.unsqueeze(-1)
                else:
                    head_mask = [None] * num_hidden_layers
                return head_mask

            if not hasattr(mixin, "_convert_head_mask_to_5d"):
                mixin._convert_head_mask_to_5d = _convert_head_mask_to_5d
            mixin.get_head_mask = get_head_mask

    def load(self):
        import torch
        self._patch_transformers_compat()
        import ImageReward as RM
        if not torch.cuda.is_available():
            self.device = "cpu"
        self.model = RM.load(self.model_id, device=self.device)
        return self

    def compute(self, record, image, prompt, reference=None):
        return float(self.model.score(prompt, [image.convert("RGB")]))

    def to_reward(self, raw: float) -> float:
        return float(1.0 / (1.0 + math.exp(-raw)))