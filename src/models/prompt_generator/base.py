"""
    Base class for the prompt-generator policy.
"""
from __future__ import annotations

import re
from abc import ABC
from pathlib import Path
from typing import Optional

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert prompt engineer for a text-to-image model. You are given "
    "the structured metadata of a finished graphic design (its canvas, colours, "
    "text content and decorative elements). Rewrite this metadata into ONE vivid, "
    "self-contained natural-language image-generation prompt that would recreate "
    "the design.\n"
    "Rules:\n"
    "- Write a single flowing paragraph, no lists, no preamble, no quotes.\n"
    "- Describe layout, composition, colour palette, style and mood concretely.\n"
    "- Include every piece of on-design text verbatim, in quotes, with where it sits.\n"
    "- Do not mention JSON, metadata, pixels, or coordinates.\n"
    "- Output only the prompt."
)


class BasePromptGenerator(ABC):
    """A LoRA-trainable HF causal-LM that maps layout text -> image prompt."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model_id: str = self.config.get("model_id", "")
        self.dtype_str: str = self.config.get("dtype", "bfloat16")
        self.device: str = self.config.get("device", "cuda")
        self.trainable: bool = bool(self.config.get("trainable", True))
        self.max_new_tokens: int = int(self.config.get("max_new_tokens", 320))
        self.temperature: float = float(self.config.get("temperature", 1.0))
        self.top_p: float = float(self.config.get("top_p", 0.95))
        self.layout_serialization: str = self.config.get("layout_serialization", "structured")
        # Reasoning ("thinking") models (e.g. Qwen3) otherwise emit a chain of
        # thought before the answer; we only want the final image prompt.
        self.enable_thinking: bool = bool(self.config.get("enable_thinking", False))
        self.system_prompt: str = self.config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        self.lora_cfg: dict = self.config.get("lora", {}) or {}
        self.model = None
        self.tokenizer = None

    # -- dtype helper ------------------------------------------------------ #
    def _torch_dtype(self):
        import torch
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}.get(self.dtype_str, torch.bfloat16)

    # -- loading ----------------------------------------------------------- #
    def load(self) -> "BasePromptGenerator":
        """Load tokenizer + model and (if trainable) attach a LoRA adapter."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ...utils import logging as log
        log.info(f"Loading prompt generator '{self.model_id}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=self._torch_dtype())
        model.to(self.device)

        if self.trainable:
            model.config.use_cache = False
            init_adapter = self.config.get("init_adapter")
            if init_adapter:
                # Resume / warm-start: continue training from a previously saved LoRA
                # adapter instead of a fresh random one. is_trainable=True keeps the
                # adapter params requires_grad so GRPO keeps optimising them (the
                # default PeftModel.from_pretrained loads for inference / frozen).
                from peft import PeftModel
                log.info(f"Attaching trainable LoRA adapter from {init_adapter}")
                model = PeftModel.from_pretrained(
                    model, str(init_adapter), is_trainable=True)
            else:
                from peft import LoraConfig, get_peft_model
                lora = LoraConfig(
                    r=int(self.lora_cfg.get("r", 16)),
                    lora_alpha=int(self.lora_cfg.get("alpha", 32)),
                    lora_dropout=float(self.lora_cfg.get("dropout", 0.0)),
                    bias="none",
                    target_modules=self.lora_cfg.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]),
                    task_type="CAUSAL_LM")
                model = get_peft_model(model, lora)
            model.print_trainable_parameters()
            # Trade compute for memory: recompute layer activations in the backward
            # pass instead of storing them. Essential here because RL sequences are
            # long (verbose layout prompts), and full-depth activations for a 9B
            # model would otherwise OOM. use_reentrant=False saves activations on a
            # frozen base with a LoRA graph without needing input grads (which would
            # break the inference_mode generation path via a requires_grad_ hook).
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.eval()
        self.model = model
        return self

    # -- prompt formatting ------------------------------------------------- #
    def build_messages(self, layout_text: str) -> list[dict]:
        """Chat messages for one layout. Override for backend-specific formatting."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",
             "content": f"Design metadata:\n{layout_text}\n\nWrite the image-generation prompt."},
        ]

    def _render_chat(self, layout_text: str) -> str:
        messages = self.build_messages(layout_text)
        # ``enable_thinking`` is understood by reasoning-model chat templates
        # (e.g. Qwen3) and harmlessly ignored by templates that don't use it.
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    def prompt_ids(self, layout_text: str):
        text = self._render_chat(layout_text)
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    @staticmethod
    def clean_completion(text: str) -> str:
        """Strip reasoning blocks, code fences and stray quotes around the prompt."""
        t = text.strip()
        # Drop any chain-of-thought: keep only what follows the last </think>,
        # then remove any remaining <think>...</think> pairs, then a dangling
        # opening <think> (a completion truncated mid-thought).
        if "</think>" in t:
            t = t.rsplit("</think>", 1)[-1].strip()
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
        t = re.sub(r"^<think>.*$", "", t, flags=re.DOTALL).strip()
        t = re.sub(r"^```[a-zA-Z]*", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
        if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
            t = t[1:-1].strip()
        return t

    # -- inference --------------------------------------------------------- #
    def generate(self, layout_text: str, sample: bool = False,
                 seed: Optional[int] = None) -> str:
        """Generate a single image prompt (used at inference)."""
        import torch
        prompt_ids = self.prompt_ids(layout_text)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens,
                          pad_token_id=self.tokenizer.pad_token_id)
        if sample:
            if seed is not None:
                torch.manual_seed(seed)
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=self.top_p)
        else:
            gen_kwargs.update(do_sample=False)
        attn = torch.ones_like(prompt_ids)
        was_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        with torch.inference_mode():
            out = self.model.generate(prompt_ids, attention_mask=attn, **gen_kwargs)
        self.model.config.use_cache = was_cache
        comp = out[0, prompt_ids.shape[1]:]
        return self.clean_completion(self.tokenizer.decode(comp, skip_special_tokens=True))

    # -- RL primitives ----------------------------------------------------- #
    def sample_group(self, prompt_ids, group_size: int):
        """Sample ``group_size`` completions for one prompt. Returns (seqs, prompt_len)."""
        import torch
        prompt_len = prompt_ids.shape[1]
        attn = torch.ones_like(prompt_ids)
        was_cache = getattr(self.model.config, "use_cache", True)
        self.model.config.use_cache = True
        with torch.inference_mode():
            gen = self.model.generate(
                prompt_ids, attention_mask=attn, do_sample=True, temperature=self.temperature,
                top_p=self.top_p, max_new_tokens=self.max_new_tokens,
                num_return_sequences=group_size, pad_token_id=self.tokenizer.pad_token_id)
        self.model.config.use_cache = was_cache
        return gen, prompt_len

    def decode(self, seq, prompt_len: int) -> str:
        return self.clean_completion(
            self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))

    def completion_logprobs(self, input_ids, attn_mask, chunk: int = 512):
        """Per-token log-probs of each token given the previous (shape [B, T-1]).

        Memory-frugal: computing the per-token log-prob as
        ``logit_label - logsumexp(logits)`` already avoids the full fp32
        ``log_softmax`` tensor, but ``out.logits[:, :-1, :].float()`` still
        materialises a whole ``[B, T-1, V]`` fp32 copy of the logits, which at
        Qwen's ~152k vocab is easy to OOM on a long sequence. We therefore
        upcast + reduce the vocab dimension in chunks along time, so the fp32
        copy is bounded to ``[B, chunk, V]`` regardless of sequence length. The
        result is numerically identical to the unchunked computation.
        """
        import torch
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        parts = []
        for t0 in range(0, logits.shape[1], chunk):
            lg = logits[:, t0:t0 + chunk, :].float()
            lab = labels[:, t0:t0 + chunk]
            tok = lg.gather(-1, lab.unsqueeze(-1)).squeeze(-1)
            parts.append(tok - torch.logsumexp(lg, dim=-1))
        return torch.cat(parts, dim=1)

    def reference_logprobs(self, input_ids, attn_mask):
        """Completion log-probs from the frozen reference (LoRA adapter disabled)."""
        import torch
        with torch.no_grad():
            if hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    return self.completion_logprobs(input_ids, attn_mask)
            return self.completion_logprobs(input_ids, attn_mask)

    # -- checkpoint IO ----------------------------------------------------- #
    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))

    def load_adapter(self, path: str | Path) -> None:
        """Load a previously trained LoRA adapter onto the base model (inference)."""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, str(path))
        self.model.eval()