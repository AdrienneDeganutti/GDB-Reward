from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterator, Optional

_PX_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _px(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    match = _PX_RE.search(str(value))
    return float(match.group()) if match else default


def _translate_xy(transform: Optional[str]) -> tuple[float, float]:
    """Extract the (x, y) of a CSS ``translate(...)`` in a transform string."""
    if not transform:
        return 0.0, 0.0
    match = re.search(r"translate\(\s*([^,]+),\s*([^)]+)\)", transform)
    if not match:
        return 0.0, 0.0
    return _px(match.group(1)), _px(match.group(2))


def _style(node: dict) -> dict:
    if not isinstance(node, dict):
        return {}
    style = node.get("style")
    if isinstance(style, dict):
        return {**node, **style}
    return node


class LayoutRecord:
    def __init__(self, layout_id: str, layout: dict, annotation: Optional[dict] = None):
        self.id = layout_id
        self.layout = layout
        self.annotation = annotation or {}

    # -- accessors --------------------------------------------------------- #
    @property
    def layout_config(self) -> dict:
        return self.layout.get("layout_config", self.layout)

    @property
    def semantic(self) -> dict:
        return self.layout.get("layout_semantic_description", {})

    @property
    def canvas(self) -> tuple[int, int]:
        meta = self.layout.get("layout_metadata", {})
        root_style = _style(self.layout_config)
        w = int(_px(meta.get("width") or root_style.get("width"), 1080))
        h = int(_px(meta.get("height") or root_style.get("height"), 1080))
        return w, h

    @property
    def user_intent(self) -> str:
        return (self.semantic.get("user_intent")
                or self.annotation.get("user_intent")
                or "").strip()

    # -- component flattening --------------------------------------------- #
    def _flatten(self) -> list[dict]:
        """Recurse the component tree into a flat, painter-ordered list."""
        out: list[dict] = []

        def walk(node: dict) -> None:
            ntype = node.get("type")
            style = _style(node)
            if ntype == "TEXT" or "text" in node:
                out.append({
                    "type": "TEXT",
                    "text": node.get("text", ""),
                    "color": style.get("color"),
                    "fontFamily": style.get("fontFamily"),
                    "fontSize": style.get("fontSize"),
                    "fontWeight": style.get("fontWeight"),
                    "textAlign": style.get("textAlign"),
                    "left": _px(style.get("left")),
                    "top": _px(style.get("top")),
                })
            elif ntype == "IMAGE":
                tx, ty = _translate_xy(style.get("transform"))
                out.append({
                    "type": "IMAGE",
                    "alt": node.get("alt", ""),
                    "width": _px(style.get("width")),
                    "height": _px(style.get("height")),
                    "left": _px(style.get("left")) + tx,
                    "top": _px(style.get("top")) + ty,
                })
            elif ntype == "GROUP":
                bg = style.get("background") or style.get("backgroundColor")
                inner = node.get("components", [])
                inner_bg = None
                for child in inner:
                    cstyle = _style(child)
                    inner_bg = inner_bg or cstyle.get("background") or cstyle.get("backgroundColor")
                color = bg or inner_bg
                if color:
                    out.append({
                        "type": "GROUP",
                        "background": color,
                        "width": _px(style.get("width")),
                        "height": _px(style.get("height")),
                    })
            for child in node.get("components", []):
                walk(child)

        walk(self.layout_config)
        return out

    def text_strings(self) -> list[str]:
        return [c["text"] for c in self._flatten() if c["type"] == "TEXT" and c["text"]]

    # -- serializers ------------------------------------------------------- #
    def serialize(self, mode: str = "structured") -> str:
        if mode == "semantic":
            return self._serialize_semantic()
        if mode == "raw_json":
            return self._serialize_raw_json()
        if mode == "structured":
            return self._serialize_structured()
        raise ValueError(f"Unknown layout_serialization mode: {mode!r}")

    def _serialize_semantic(self) -> str:
        sem = self.semantic or self.annotation
        parts = []
        for key in ("description", "aesthetics", "tags", "user_intent"):
            val = sem.get(key)
            if val:
                parts.append(f"{key.replace('_', ' ').title()}: {val}")
        return "\n\n".join(parts) if parts else self._serialize_structured()

    def _serialize_raw_json(self) -> str:
        trimmed = {
            "canvas": self.canvas,
            "background": _style(self.layout_config).get("background"),
            "components": self._flatten(),
        }
        return json.dumps(trimmed, ensure_ascii=False, indent=2)

    def _serialize_structured(self) -> str:
        w, h = self.canvas
        root_bg = _style(self.layout_config).get("background")
        lines = [f"Canvas: {w}x{h} px.",
                 f"Background color: {root_bg or 'unspecified'}.", ""]

        comps = self._flatten()
        texts = [c for c in comps if c["type"] == "TEXT" and c["text"]]
        images = [c for c in comps if c["type"] == "IMAGE" and c["alt"]]
        groups = [c for c in comps if c["type"] == "GROUP"]

        if texts:
            lines.append("Text elements (in reading order):")
            for t in texts:
                desc = f'  - "{t["text"]}"'
                meta = []
                if t.get("fontSize"):
                    meta.append(f"size {t['fontSize']}")
                if t.get("fontFamily"):
                    meta.append(f"font {t['fontFamily']}")
                if t.get("color"):
                    meta.append(f"color {t['color']}")
                if t.get("textAlign"):
                    meta.append(f"{t['textAlign']}-aligned")
                if meta:
                    desc += f" ({', '.join(meta)})"
                lines.append(desc)
            lines.append("")

        if groups:
            colors = [g["background"] for g in groups if g.get("background")]
            if colors:
                lines.append("Color blocks / shapes: " + "; ".join(colors[:8]) + ".")
                lines.append("")

        if images:
            lines.append("Decorative graphic elements:")
            for im in images[:24]:
                lines.append(f"  - {im['alt'].strip()}")
            lines.append("")

        return "\n".join(lines).strip()


def _scan_layout_paths(root: Path) -> dict[str, Path]:
    """Map layout id -> JSON path, supporting flat and folder-per-layout layouts.

    Two on-disk conventions are handled (and may coexist):
      * **flat**   : ``<root>/<layout_id>.json`` (id = file stem).
      * **nested** : ``<root>/<layout_id>/<any>.json`` (id = folder name), which
        is how the LICA dataset ships (``layouts/<uuid>/<inner>.json``). If a
        folder holds several JSONs the first (sorted) is used.

    Both are scanned and merged, so a stray top-level file (e.g. LICA's
    ``annotations/template_annotations.json``) never hides the per-layout
    folders. Folder ids win on the rare collision.
    """
    paths: dict[str, Path] = {}
    for p in sorted(root.glob("*.json")):
        paths[p.stem] = p
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        inner = sorted(sub.glob("*.json"))
        if inner:
            paths[sub.name] = inner[0]
    return paths


def _load_split_csv(csv_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    root = csv_path.parent
    layouts: dict[str, Path] = {}
    annotations: dict[str, Path] = {}
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            lid = (row.get("layout_id") or "").strip()
            if not lid:
                continue
            lp = (row.get("layout_path") or "").strip()
            if lp:
                p = Path(lp)
                layouts[lid] = p if p.is_absolute() else (root / p)
            ap = (row.get("annotation_path") or "").strip()
            if ap:
                p = Path(ap)
                annotations[lid] = p if p.is_absolute() else (root / p)
    return layouts, annotations


class LayoutDataset:
    def __init__(self, layouts_dir: str | Path, annotations_dir: Optional[str | Path] = None,
                 split_csv: Optional[str | Path] = None):
        self.layouts_dir = Path(layouts_dir)
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None

        if split_csv is not None:
            split_csv = Path(split_csv)
            if not split_csv.is_file():
                raise FileNotFoundError(f"split_csv not found: {split_csv}")
            self._layout_paths, self._annotation_paths = _load_split_csv(split_csv)
            if not self._layout_paths:
                raise FileNotFoundError(f"No layouts listed in split_csv: {split_csv}")
        else:
            if not self.layouts_dir.is_dir():
                raise FileNotFoundError(f"layouts_dir not found: {self.layouts_dir}")
            self._layout_paths = _scan_layout_paths(self.layouts_dir)
            if not self._layout_paths:
                raise FileNotFoundError(f"No layout JSON files in {self.layouts_dir}")
            self._annotation_paths = (
                _scan_layout_paths(self.annotations_dir)
                if self.annotations_dir and self.annotations_dir.is_dir() else {})
        self.ids = sorted(self._layout_paths)

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[LayoutRecord]:
        for layout_id in self.ids:
            yield self.get(layout_id)

    def get(self, layout_id: str) -> LayoutRecord:
        layout_path = self._layout_paths.get(layout_id)
        if layout_path is None:
            raise KeyError(f"Unknown layout id: {layout_id!r}")
        layout = json.loads(layout_path.read_text())
        annotation = None
        ann_path = self._annotation_paths.get(layout_id)
        if ann_path is not None:
            annotation = json.loads(ann_path.read_text())
        return LayoutRecord(layout_id, layout, annotation)

    def __getitem__(self, index: int) -> LayoutRecord:
        return self.get(self.ids[index])