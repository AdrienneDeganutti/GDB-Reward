"""Zero-dependency colour helpers shared by the colour/palette metrics.

CSS colour parsing, layout target-colour extraction, dominant-palette
extraction, sRGB->Lab conversion, CIEDE2000 perceptual distance, and WCAG
relative luminance. These are used by ``color_deltae`` and depend only on
numpy + Pillow.
"""
from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

from PIL import Image

from .base import RGB

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...utils.data import LayoutRecord

_HEX_RE = re.compile(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
_RGB_RE = re.compile(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", re.IGNORECASE)


def parse_colors(value: object) -> list[RGB]:
    """Extract every solid colour from a CSS string (handles gradients too)."""
    if not isinstance(value, str):
        return []
    out: list[RGB] = []
    for m in _HEX_RE.finditer(value):
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        out.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
    for m in _RGB_RE.finditer(value):
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def _node_style(node: dict) -> dict:
    style = node.get("style")
    return style if isinstance(style, dict) else {}


def layout_target_colors(record: "LayoutRecord") -> list[RGB]:
    """Gather the colours the design is *supposed* to contain.

    Pulls the root background plus every text colour and group/shape fill from
    the layout's component tree. These are the ground-truth colours the
    generated image is scored against.

    LICA layouts carry style attributes *flat* on each component (``color``,
    ``background`` directly on the node), while some exports nest them under a
    ``style`` dict. We read both so the metric works across formats.
    """
    colors: list[RGB] = []
    lc = record.layout_config
    colors += parse_colors(lc.get("background"))
    colors += parse_colors(_node_style(lc).get("background"))

    def walk(node: dict) -> None:
        if not isinstance(node, dict):
            return
        style = _node_style(node)
        ntype = node.get("type")
        if ntype == "TEXT" or "text" in node:
            colors.extend(parse_colors(node.get("color") or style.get("color")))
        if ntype == "GROUP" or "background" in node or "backgroundColor" in node:
            colors.extend(parse_colors(
                node.get("background") or node.get("backgroundColor")
                or style.get("background") or style.get("backgroundColor")))
        for child in node.get("components", []):
            walk(child)

    walk(lc)

    # de-duplicate, preserve order
    seen: set[RGB] = set()
    uniq: list[RGB] = []
    for c in colors:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def image_palette(image: Image.Image, k: int = 6) -> list[tuple[RGB, float]]:
    """Dominant colours of an image as ``[(rgb, weight), ...]`` (weights sum 1)."""
    im = image.convert("RGB").resize((128, 128))
    q = im.quantize(colors=max(2, k), method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette() or []
    counts = q.getcolors(maxcolors=128 * 128) or []
    out: list[tuple[RGB, int]] = []
    for count, idx in counts:
        rgb = (pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2])
        out.append((rgb, count))
    total = sum(c for _, c in out) or 1
    palette = [(rgb, c / total) for rgb, c in out]
    palette.sort(key=lambda t: -t[1])
    return palette


def _srgb_to_lab(rgb: RGB) -> tuple[float, float, float]:
    def lin(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(rgb[0]), lin(rgb[1]), lin(rgb[2]))
    # sRGB -> XYZ (D65)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 1.00000
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def ciede2000(rgb1: RGB, rgb2: RGB) -> float:
    """Perceptual ΔE (CIEDE2000) between two sRGB colours (Sharma et al.)."""
    l1, a1, b1 = _srgb_to_lab(rgb1)
    l2, a2, b2 = _srgb_to_lab(rgb2)
    avg_lp = (l1 + l2) / 2
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    avg_c = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(avg_c ** 7 / (avg_c ** 7 + 25 ** 7))) if avg_c > 0 else 0.0
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    avg_cp = (c1p + c2p) / 2

    def hp(ap: float, bp: float) -> float:
        if ap == 0 and bp == 0:
            return 0.0
        h = math.degrees(math.atan2(bp, ap))
        return h + 360 if h < 0 else h

    h1p, h2p = hp(a1p, b1), hp(a2p, b2)
    dlp = l2 - l1
    dcp = c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp) / 2)

    if c1p * c2p == 0:
        avg_hp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        avg_hp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        avg_hp = (h1p + h2p + 360) / 2
    else:
        avg_hp = (h1p + h2p - 360) / 2

    t = (1 - 0.17 * math.cos(math.radians(avg_hp - 30))
         + 0.24 * math.cos(math.radians(2 * avg_hp))
         + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
         - 0.20 * math.cos(math.radians(4 * avg_hp - 63)))
    sl = 1 + (0.015 * (avg_lp - 50) ** 2) / math.sqrt(20 + (avg_lp - 50) ** 2)
    sc = 1 + 0.045 * avg_cp
    sh = 1 + 0.015 * avg_cp * t
    dtheta = 30 * math.exp(-(((avg_hp - 275) / 25) ** 2))
    rc = 2 * math.sqrt(avg_cp ** 7 / (avg_cp ** 7 + 25 ** 7)) if avg_cp > 0 else 0.0
    rt = -rc * math.sin(math.radians(2 * dtheta))
    return math.sqrt(
        (dlp / sl) ** 2 + (dcp / sc) ** 2 + (dHp / sh) ** 2
        + rt * (dcp / sc) * (dHp / sh))


def relative_luminance(rgb: RGB) -> float:
    """WCAG relative luminance of an sRGB colour in ``[0, 1]``."""
    def lin(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(rgb[0]), lin(rgb[1]), lin(rgb[2])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b