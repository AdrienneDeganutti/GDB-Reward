"""GRPO fine-tuning of the prompt-generator LLM.

One step of the loop, for a sampled layout:
  1. serialize the layout metadata and sample a GROUP of G image prompts from the
     policy (LoRA on top of the base LLM);
  2. for each prompt, generate an image with the (frozen) image generator and
     score it with the (frozen) reward -> a scalar reward per sample;
  3. group-relative advantage  A_i = (r_i - mean) / (std + eps);
  4. REINFORCE on the completion tokens with a KL penalty to the frozen reference
     policy (the same model with the LoRA adapter disabled).

The default reward is the GraphicDesignBench composite ``GDBReward``
(``reward.name: gdb``; see ``src/tasks/gdb_reward``), which scores each generated
image against the source layout and folds a weighted mix of metrics into a scalar
in ``[0, 1]``. ``reward.name: placeholder`` swaps in a non-learning stand-in.
"""
from __future__ import annotations

import gc
import json
import os
import random
import re
import time
from pathlib import Path

import torch

from ..models import build_image_generator, build_prompt_generator
from ..utils import logging as log
from ..utils.config import Config
from ..utils.data import LayoutDataset
from ..utils.logging import JsonlLogger
from ..utils.reward import build_reward


def _resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path)


_STEP_RE = re.compile(r"step_(\d+)$")


def _step_of(path: Path) -> int:
    """Parse the completed-step count N from a ``step_<N>`` checkpoint dir name."""
    m = _STEP_RE.search(path.name)
    if not m:
        raise ValueError(
            f"cannot parse a step number from checkpoint dir '{path}' "
            "(expected a 'step_<N>' directory).")
    return int(m.group(1))


def _latest_checkpoint(out_dir: Path) -> Path:
    """Return the ``step_<N>`` dir with the highest N under ``out_dir``."""
    ckpts = [p for p in out_dir.glob("step_*") if p.is_dir() and _STEP_RE.search(p.name)]
    if not ckpts:
        raise FileNotFoundError(
            f"grpo.resume=true but no 'step_<N>' checkpoint was found in {out_dir}.")
    return max(ckpts, key=_step_of)


def _apply_cuda_visible_devices(cfg: Config) -> None:
    """Pin the run to specific GPUs before torch/CUDA is initialised.

    Accepts an int (``0``), a list (``[0, 1]``) or a string (``"0,1"``) under the
    top-level ``cuda_visible_devices`` key. Must run before any CUDA call, as
    ``CUDA_VISIBLE_DEVICES`` only takes effect prior to CUDA initialisation.
    """
    # Reduce allocator fragmentation (helps when several large models share a GPU);
    # must be set before CUDA initialises. `setdefault` so the user can override.
    # PYTORCH_ALLOC_CONF is the current name (PYTORCH_CUDA_ALLOC_CONF is deprecated).
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    cvd = cfg.get("cuda_visible_devices")
    if cvd is None or cvd == "":
        return
    value = ",".join(str(x) for x in cvd) if isinstance(cvd, (list, tuple)) else str(cvd)
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    log.info(f"CUDA_VISIBLE_DEVICES={value}")


def _init_wandb(cfg: Config, out_dir: Path):
    """Start a Weights & Biases run from the ``wandb`` config block, if enabled.

    Returns the run handle, or ``None`` when logging is disabled or the ``wandb``
    package is not installed (training never hard-fails on logging). Config keys
    (all optional): ``enabled`` (default false), ``project``, ``entity``,
    ``name``, ``group``, ``tags``, ``mode`` (online | offline | disabled).
    """
    wcfg = cfg.get("wandb") or {}
    if not bool(wcfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError:
        log.warn("wandb enabled in config but not installed; skipping "
                 "(pip install wandb).")
        return None
    run = wandb.init(
        project=wcfg.get("project", "dedesc-grpo"),
        entity=wcfg.get("entity"),
        name=wcfg.get("name"),
        group=wcfg.get("group"),
        tags=list(wcfg.get("tags") or []),
        mode=wcfg.get("mode", "online"),
        dir=str(out_dir),
        config=cfg.to_dict(),
    )
    log.info(f"wandb: run '{run.name}' -> {run.url}")
    return run


def _validate(*, step: int, policy, image_gen, reward_fn, dataset, serialization: str,
              out_dir: Path, seed: int, max_layouts: int, do_sample: bool,
              do_score: bool, wandb_run, val_logger) -> dict:
    """Run the current policy over the validation split and dump its outputs.

    For each layout: generate one prompt (greedy by default, so the val curve is
    comparable across checkpoints), render an image, and save both. Writes to
    ``<out_dir>/val/step_<N>/{prompts/<id>.txt, images/<id>.png}`` plus a
    ``val_results.json`` index, and appends a summary row to ``grpo_val.jsonl``.

    When ``do_score`` is set (default) the frozen reward also scores each image so
    a validation reward curve (and its per-metric decomposition) is logged; the
    generated images and prompts are always produced regardless.
    """
    tag = step + 1
    vdir = out_dir / "val" / f"step_{tag}"
    vprompts, vimages = vdir / "prompts", vdir / "images"
    vprompts.mkdir(parents=True, exist_ok=True)
    vimages.mkdir(parents=True, exist_ok=True)

    ids = dataset.ids
    if max_layouts and max_layouts < len(ids):
        ids = ids[:max_layouts]  # deterministic subset (ids are sorted) -> comparable across steps

    policy.model.eval()
    t0 = time.time()
    results, metric_vals, reward_sum, scored = [], {}, 0.0, 0
    for lid in ids:
        record = dataset.get(lid)
        layout_text = record.serialize(serialization)
        prompt = policy.generate(layout_text, sample=do_sample,
                                 seed=(seed if do_sample else None))
        (vprompts / f"{lid}.txt").write_text(prompt)
        image = image_gen.generate(prompt, seed=seed)
        img_path = vimages / f"{lid}.png"
        image.save(img_path)
        entry = {"layout_id": lid, "prompt": prompt, "image": str(img_path)}
        if do_score:
            rinfo = reward_fn.score(record, image, prompt)
            entry["reward"] = float(rinfo["reward"])
            reward_sum += entry["reward"]
            scored += 1
            for k, v in rinfo.items():
                if k.endswith("_r") and v is not None:
                    metric_vals.setdefault(k[:-2], []).append(float(v))
        results.append(entry)

    (vdir / "val_results.json").write_text(json.dumps(results, indent=2))

    summary = {"step": step, "n": len(results), "sec": round(time.time() - t0, 1)}
    if scored:
        summary["mean_reward"] = round(reward_sum / scored, 4)
        for mk, vals in metric_vals.items():
            summary[f"reward_{mk}"] = round(sum(vals) / len(vals), 4)
    echo = f"[val] step {step}: {len(results)} layouts -> {vdir}"
    if scored:
        echo += f" | mean_reward={summary['mean_reward']:.3f}"
    echo += f" ({summary['sec']}s)"
    if val_logger is not None:
        val_logger.log(summary, echo=echo)
    if wandb_run is not None and scored:
        wlog = {"val/mean_reward": summary["mean_reward"]}
        for mk, vals in metric_vals.items():
            wlog[f"val/{mk}"] = sum(vals) / len(vals)
        wandb_run.log(wlog, step=step)

    policy.model.train()
    return summary


def run(cfg: Config) -> None:
    _apply_cuda_visible_devices(cfg)  # must precede any CUDA initialisation
    project_root = Path(cfg.get("_config_path", ".")).resolve().parents[2] \
        if cfg.get("_config_path") else Path.cwd()

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)

    g = cfg.grpo
    out_dir = _resolve_path(project_root, cfg.get("output_dir", "outputs/grpo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = out_dir / "renders"
    if bool(g.get("save_images", False)):
        renders_dir.mkdir(exist_ok=True)

    split_csv = cfg.data.get("split_csv")
    dataset = LayoutDataset(
        _resolve_path(project_root, cfg.data.layouts_dir),
        _resolve_path(project_root, cfg.data.annotations_dir)
        if cfg.data.get("annotations_dir") else None,
        split_csv=_resolve_path(project_root, split_csv) if split_csv else None)
    serialization = cfg.prompt_generator.get("layout_serialization", "structured")
    log.info(f"Loaded {len(dataset)} layouts from {cfg.data.layouts_dir}")

    val_split = cfg.data.get("val_split_csv")
    val_dataset = LayoutDataset(
        _resolve_path(project_root, cfg.data.layouts_dir),
        _resolve_path(project_root, cfg.data.annotations_dir)
        if cfg.data.get("annotations_dir") else None,
        split_csv=_resolve_path(project_root, val_split)) if val_split else None
    if val_dataset is not None:
        log.info(f"Validation split: {len(val_dataset)} layouts from {val_split}")

    start_step = 0
    resume_from = g.get("resume_from")
    if resume_from:
        resume_dir = _resolve_path(project_root, resume_from)
    elif bool(g.get("resume", False)):
        resume_dir = _latest_checkpoint(out_dir)
    else:
        resume_dir = None

    pg_cfg = cfg.prompt_generator.to_dict()
    if resume_dir is not None:
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_dir}")
        start_step = _step_of(resume_dir)
        pg_cfg["init_adapter"] = str(resume_dir)
        log.info(f"Resuming from {resume_dir} -> starting at step {start_step}/{int(g.steps)}")

    # --- models (policy trainable; generator + reward frozen) ------------- #
    policy = build_prompt_generator(pg_cfg).load()
    image_gen = build_image_generator(cfg.image_generator.to_dict()).load()
    reward_cfg = cfg.reward.to_dict()
    if reward_cfg.get("renders_dir"):
        reward_cfg["renders_dir"] = str(
            _resolve_path(project_root, reward_cfg["renders_dir"]))
    reward_fn = build_reward(reward_cfg).load()

    device = next(policy.model.parameters()).device
    log.info(f"policy device: {device}")

    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=float(g.lr))

    steps = int(g.steps)
    group_size = int(g.group_size)
    layouts_per_step = int(g.get("layouts_per_step", 1))
    micro_batch = int(g.get("micro_batch", 2))
    kl_coeff = float(g.get("kl_coeff", 0.04))
    adv_eps = float(g.get("adv_eps", 1e-4))
    grad_clip = float(g.get("grad_clip", 1.0))
    save_every = int(g.get("save_every", 25))
    save_prompts = bool(g.get("save_prompts", True))
    # Common-random-numbers: share ONE diffusion seed across every member of a group
    # (still varies across steps). Within a group the members then differ only by the
    # prompt (the policy's action), so the group-relative advantage isolates prompt
    # quality instead of being dominated by per-sample seed noise. Default False keeps
    # the original per-sample-seed behaviour.
    shared_group_seed = bool(g.get("shared_group_seed", False))
    # Validation cadence (in steps). One pass over the train split is
    # ceil(len(train) / layouts_per_step) steps, so set `val_every` to that for
    # per-"epoch" validation. 0 (or no val split) disables it.
    val_every = int(g.get("val_every", 0) or 0)
    val_max_layouts = int(g.get("val_max_layouts", 0) or 0)  # 0 = full split
    val_sample = bool(g.get("val_sample", False))            # False -> greedy, comparable curve
    val_score = bool(g.get("val_score", True))               # also score val images with the reward
    pad_id = policy.tokenizer.pad_token_id

    logger = JsonlLogger(out_dir, "grpo_log")
    prompt_logger = JsonlLogger(out_dir, "grpo_prompts") if save_prompts else None
    val_logger = (JsonlLogger(out_dir, "grpo_val")
                  if val_dataset is not None and val_every > 0 else None)
    wandb_run = _init_wandb(cfg, out_dir)
    log.info(f"GRPO: {steps} steps | G={group_size} | layouts/step={layouts_per_step}"
             + (f" | resuming at step {start_step}" if start_step else ""))

    for step in range(start_step, steps):
        t0 = time.time()
        step_records = random.sample(dataset.ids, min(layouts_per_step, len(dataset)))
        batch = []

        for layout_id in step_records:
            record = dataset.get(layout_id)
            layout_text = record.serialize(serialization)
            prompt_ids = policy.prompt_ids(layout_text)

            # 1) sample a group of prompt completions
            policy.model.eval()
            seqs, prompt_len = policy.sample_group(prompt_ids, group_size)

            # 2) reward each: prompt -> image -> score. Generate the whole group in
            # one batched pipeline call (one denoising loop over G prompts) rather
            # than G separate calls -- a large speedup for the frozen generator.
            prompt_texts = [policy.decode(seqs[gi], prompt_len) for gi in range(group_size)]
            if shared_group_seed:
                # same seed for all G members -> reward differences reflect the prompt,
                # not the diffusion noise (common random numbers within the group).
                seeds_g = [seed * 100003 + step] * group_size
            else:
                seeds_g = [seed * 100003 + step * group_size + gi for gi in range(group_size)]
            images = image_gen.generate_batch(prompt_texts, seeds=seeds_g)
            rewards, samples = [], []
            for gi in range(group_size):
                prompt_text = prompt_texts[gi]
                image = images[gi]
                if bool(g.get("save_images", False)):
                    image.save(renders_dir / f"s{step}_{layout_id}_g{gi}.png")
                rinfo = reward_fn.score(record, image, prompt_text)
                rewards.append(float(rinfo["reward"]))
                samples.append({"prompt": prompt_text, **{k: v for k, v in rinfo.items()
                                                          if k != "reward"}})
            batch.append({"layout_id": layout_id, "seqs": seqs, "prompt_len": prompt_len,
                          "rewards": rewards, "samples": samples})

        # 3) group-relative advantages
        flat_seqs, flat_adv, flat_plen = [], [], []
        all_rewards = []
        for rec in batch:
            rs = torch.tensor(rec["rewards"], dtype=torch.float32)
            adv = (rs - rs.mean()) / (rs.std() + adv_eps)
            rec["advantages"] = adv.tolist()
            all_rewards.extend(rec["rewards"])
            for gi in range(group_size):
                flat_seqs.append(rec["seqs"][gi])
                flat_adv.append(adv[gi].item())
                flat_plen.append(rec["prompt_len"])

        # 4) policy-gradient update with KL to the frozen reference
        policy.model.train()
        optimizer.zero_grad()
        pg_total, kl_total, ntok = 0.0, 0.0, 0.0
        for i in range(0, len(flat_seqs), micro_batch):
            chunk = flat_seqs[i:i + micro_batch]
            advs = flat_adv[i:i + micro_batch]
            plens = flat_plen[i:i + micro_batch]
            maxlen = max(s.shape[0] for s in chunk)
            ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long, device=device)
            attn = torch.zeros((len(chunk), maxlen), dtype=torch.long, device=device)
            for j, s in enumerate(chunk):
                ids[j, :s.shape[0]] = s
                attn[j, :s.shape[0]] = 1

            tok_logp = policy.completion_logprobs(ids, attn)
            with torch.no_grad():
                ref_logp = policy.reference_logprobs(ids, attn)

            # mask = completion tokens (after each row's own prompt) and not padding
            row_mask = torch.zeros_like(tok_logp)
            for j, pl in enumerate(plens):
                row_mask[j, pl - 1:] = 1.0
            row_mask = row_mask * attn[:, 1:].float()

            adv_t = torch.tensor(advs, device=device, dtype=tok_logp.dtype).unsqueeze(1)
            logratio = ref_logp - tok_logp
            kl = torch.exp(logratio) - 1 - logratio          # k3 estimator, >= 0
            pg = -(adv_t * tok_logp)
            denom = row_mask.sum().clamp(min=1.0)
            loss = ((pg + kl_coeff * kl) * row_mask).sum() / denom
            loss.backward()

            pg_total += (pg * row_mask).sum().item()
            kl_total += (kl * row_mask).sum().item()
            ntok += denom.item()

        torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), grad_clip)
        optimizer.step()

        mean_r = sum(all_rewards) / len(all_rewards)
        stats = {
            "step": step,
            "mean_reward": round(mean_r, 4),
            "max_reward": round(max(all_rewards), 4),
            "reward_std": round(float(torch.tensor(all_rewards).std()), 4),
            "kl": round(kl_total / max(ntok, 1.0), 5),
            "pg": round(pg_total / max(ntok, 1.0), 5),
            "sec": round(time.time() - t0, 1),
            "example_prompt": batch[0]["samples"][0]["prompt"][:240],
        }
        logger.log(stats,
                   echo=(f"step {step}: reward={stats['mean_reward']:.3f} "
                         f"(max {stats['max_reward']:.3f}, std {stats['reward_std']:.3f}) "
                         f"kl={stats['kl']:.4f} {stats['sec']}s"))

        if wandb_run is not None:
            # Mean reward contributed by each metric across the step's samples,
            # so the composite can be decomposed on the wandb dashboard.
            metric_rewards: dict[str, list[float]] = {}
            for rec in batch:
                for smp in rec["samples"]:
                    for k, v in smp.items():
                        if k.endswith("_r") and v is not None:
                            metric_rewards.setdefault(k[:-2], []).append(float(v))
            wlog = {f"train/{k}": v for k, v in stats.items()
                    if k not in ("step", "example_prompt")}
            for mk, vals in metric_rewards.items():
                wlog[f"reward/{mk}"] = sum(vals) / len(vals)
            wandb_run.log(wlog, step=step)

        # Full per-step record: every group member's prompt, reward and advantage.
        if prompt_logger is not None:
            prompt_logger.log({
                "step": step,
                "layouts": [{
                    "layout_id": rec["layout_id"],
                    "samples": [{
                        "group_index": gi,
                        "prompt": rec["samples"][gi]["prompt"],
                        "reward": rec["rewards"][gi],
                        "advantage": rec["advantages"][gi],
                        **{k: v for k, v in rec["samples"][gi].items() if k != "prompt"},
                    } for gi in range(group_size)],
                } for rec in batch],
            })

        if (step + 1) % save_every == 0 or step == steps - 1:
            save_dir = out_dir / f"step_{step + 1}"
            policy.save(save_dir)
            log.info(f"saved LoRA -> {save_dir}")

        if val_dataset is not None and val_every > 0 and (
                (step + 1) % val_every == 0 or step == steps - 1):
            _validate(step=step, policy=policy, image_gen=image_gen, reward_fn=reward_fn,
                      dataset=val_dataset, serialization=serialization, out_dir=out_dir,
                      seed=seed, max_layouts=val_max_layouts, do_sample=val_sample,
                      do_score=val_score, wandb_run=wandb_run, val_logger=val_logger)

        del batch, flat_seqs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.save(out_dir / "final")
    logger.close()
    if prompt_logger is not None:
        prompt_logger.close()
        log.info(f"Per-step prompts -> {prompt_logger.path}")
    if val_logger is not None:
        val_logger.close()
        log.info(f"Validation summaries -> {val_logger.path}  |  outputs under {out_dir / 'val'}")
    if wandb_run is not None:
        wandb_run.finish()
    log.info(f"Done. Final policy LoRA: {out_dir / 'final'}  |  log: {logger.path}")