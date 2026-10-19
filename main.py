from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config  # noqa: E402


def _add_common(p: argparse.ArgumentParser, default_config: str) -> None:
    p.add_argument("--config", default=default_config, help="Path to a YAML config file.")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="key.subkey=value",
                   help="Override any config value (repeatable).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="GRPO fine-tune the prompt-generator LLM.")
    _add_common(p_train, "src/configs/train_config.yaml")

    p_infer = sub.add_parser("infer", help="Run layout -> prompt -> image.")
    _add_common(p_infer, "src/configs/infer_config.yaml")
    p_infer.add_argument("--layout-id", default=None,
                         help="Single layout id to run (default: all layouts).")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.overrides)

    if args.command == "train":
        from src.tasks import train
        train.run(cfg)
    elif args.command == "infer":
        from src.tasks import inference
        inference.run(cfg, layout_id=args.layout_id)


if __name__ == "__main__":
    main()