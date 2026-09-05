"""Image rendering tools for KiCad schematics, PCBs, symbols, and footprints.

Provides a single ``render_image`` tool that produces raster / vector images
suitable for passing to a vision-capable LLM for sanity-checking.  Supports
five render targets via a ``kind`` discriminator:

* ``schematic`` — render a ``.kicad_sch`` page
* ``pcb`` — render a ``.kicad_pcb`` board
* ``symbol`` — render a library symbol
* ``footprint`` — render a library footprint
* ``region`` — render a bbox-cropped region of a schematic or PCB

The tool always writes its output to disk (under ``kcaa_data_dir/render/``
by default) and returns the image bytes as MCP ``Image`` content so the
caller can pipe them directly to a vision model.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.utilities.types import Image
# Lazy: PIL not required at MCP startup; only needed for image rendering
    # Falls back gracefully if Pillow is unavailable on the system.
    # from PIL import Image as PILImage  # moved into _resize_image

from kcaa.utils.config import config
from kcaa.utils.kicad_cli import get_kicad_cli_path, KiCadCLIError
from kcaa.utils.pcb_board_utils import get_edge_cuts_items
from kcaa.utils.pcb_sexp_utils import load_pcb
from kcaa.utils.schematic_sexp_utils import load_schematic
from kcaa.utils.secure_subprocess import run_kicad_command_async

log = logging.getLogger(__name__)

log = logging.getLogger(__name__)


# Standard KiCad paper sizes (mm, landscape orientation).
_PAPER_SIZES: dict[str, tuple[float, float]] = {
    "A0": (1189.0, 841.0),
    "A1": (841.0, 594.0),
    "A2": (594.0, 420.0),
    "A3": (420.0, 297.0),
    "A4": (297.0, 210.0),
    "A5": (210.0, 148.0),
    "Letter": (279.4, 215.9),
    "Legal": (355.6, 215.9),
    "Tabloid": (431.8, 279.4),
}
_DEFAULT_PAPER = "A4"

# Layers shown by the existing ``generate_pcb_thumbnail`` tool — kept in sync.
_DEFAULT_PCB_LAYERS = (
    "F.Cu,B.Cu,F.SilkS,B.SilkS,F.Mask,B.Mask,Edge.Cuts"
)


# ---------------------------------------------------------------------------
# Source-extent helpers (mm bbox of the source document)
# ---------------------------------------------------------------------------


def _schematic_page_mm(schematic_path: str) -> tuple[float, float]:
    """Return ``(width_mm, height_mm)`` for a ``.kicad_sch`` page.

    Reads the ``(paper "...")`` entry from the schematic S-expression and
    maps it through ``_PAPER_SIZES``.  Falls back to A4 landscape if the
    paper entry is missing or unrecognised.
    """
    try:
        tree = load_schematic(schematic_path)
        for item in _walk(tree):
            if (
                isinstance(item, list)
                and len(item) >= 2
                and item[0] == "paper"
                and isinstance(item[1], str)
                and item[1] in _PAPER_SIZES
            ):
                return _PAPER_SIZES[item[1]]
    except (OSError, ValueError) as e:
        log.warning("could not parse paper size from %s: %s", schematic_path, e)
    return _PAPER_SIZES[_DEFAULT_PAPER]


def _pcb_bbox_mm(pcb_path: str) -> tuple[float, float, float, float]:
    """Return ``(x_min, y_min, x_max, y_max)`` mm bbox of Edge.Cuts on a ``.kicad_pcb``.

    Edge.Cuts contains the board outline as ``gr_line`` / ``gr_arc`` /
    ``gr_rect`` / ``gr_circle`` items.  Falls back to ``(0, 0, 100, 100)`` if
    the file has no Edge.Cuts geometry (or cannot be parsed).
    """
    try:
        data = load_pcb(pcb_path)
        items = get_edge_cuts_items(data)
        xs: list[float] = []
        ys: list[float] = []
        for it in items:
            kind = it.get("type")
            if kind in ("gr_line", "gr_rect"):
                xs += [it["x1"], it["x2"]]
                ys += [it["y1"], it["y2"]]
            elif kind == "gr_arc":
                xs += [it["start_x"], it["mid_x"], it["end_x"]]
                ys += [it["start_y"], it["mid_y"], it["end_y"]]
            elif kind == "gr_circle":
                xs += [it["cx"], it["ex"]]
                ys += [it["cy"], it["ey"]]
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
    except (OSError, ValueError, KeyError) as e:
        log.warning("could not parse Edge.Cuts bbox from %s: %s", pcb_path, e)
    return (0.0, 0.0, 100.0, 100.0)


def _walk(node):
    """Depth-first iteration over a sexpdata-style nested list."""
    if isinstance(node, list):
        yield node
        for child in node:
            yield from _walk(child)


# ---------------------------------------------------------------------------
# kicad-cli invocation helpers
# ---------------------------------------------------------------------------


async def _run_cli(
    args: list[str],
    output_path: str,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run ``kicad-cli <args> --output <output_path>`` and return the result.

    Uses the secure subprocess wrapper so input paths, output directories,
    and the executable are validated.
    """
    full_args = list(args) + ["--output", output_path]
    return await run_kicad_command_async(
        command_args=full_args,
        output_files=[output_path],
        timeout=timeout,
    )


def _load_image(
    path: str,
    fmt: Literal["png", "svg", "pdf"],
) -> tuple[bytes, PILImage.Image | None]:
    """Read ``path`` and return ``(raw_bytes, PIL_image_or_None)``.

    PIL can decode PNG natively.  SVG is returned as raw bytes only
    (Pillow cannot decode SVG without external renderers).  PDF is
    returned as raw bytes only.
    """
    with open(path, "rb") as f:
        data = f.read()
    pil: PILImage.Image | None = None
    if fmt == "png":
        try:
            opened = PILImage.open(path)
            opened.load()  # force decode so the file handle can be released
            pil = opened
        except Exception as e:  # pragma: no cover — defensive
            log.warning("Pillow could not decode %s: %s", path, e)
            pil = None
    return data, pil


def _svg_to_png(
    svg_path: str,
    png_path: str,
    output_width: int | None = None,
    background_color: str | None = None,
) -> None:
    """Convert an SVG file to PNG using a pure-Python renderer (svglib + Pillow).

    No native dependencies (no libcairo, no GTK, no ImageMagick). svglib parses
    the SVG into a reportlab Drawing tree; we walk that tree and emit
    Pillow draw calls. Handles the elements KiCad emits: rect, line,
    circle, ellipse, polygon, polyline, path (M/L/H/V/Z), text, and groups
    with translate/scale transforms.

    ``output_width`` scales the SVG so its rendered width matches the
    given pixel count (height auto from aspect).  ``background_color``
    accepts CSS-style values (``"#FFFFFF"``, ``"white"``); ``None`` is
    transparent.
    """
    # Lazy imports — these are pure-Python and not required at MCP startup
    from svglib.svglib import svg2rlg
    from reportlab.graphics.shapes import (
        Rect, Line, Circle, Ellipse, Polygon, PolyLine, String, Group, Path,
    )
    from PIL import Image, ImageDraw, ImageFont
    import re

    drawing = svg2rlg(svg_path)
    src_w = float(drawing.width) or 1.0
    src_h = float(drawing.height) or 1.0

    scale = 1.0
    if output_width is not None and output_width > 0:
        scale = float(output_width) / src_w
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))

    if background_color and background_color.lower() in ("white", "#ffffff", "#fff"):
        bg_rgba = (255, 255, 255, 255)
    elif background_color and background_color.startswith("#"):
        h = background_color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        bg_rgba = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
    else:
        bg_rgba = (255, 255, 255, 0)  # transparent

    img = Image.new("RGBA", (out_w, out_h), bg_rgba)
    draw = ImageDraw.Draw(img)

    # Try common system fonts, fall back to PIL default
    font_cache: dict[int, object] = {}
    for path, size in [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12),
        ("/usr/share/fonts/TTF/DejaVuSans.ttf", 12),
    ]:
        try:
            font_cache[12] = ImageFont.truetype(path, size)
            break
        except OSError:
            continue
    if not font_cache:
        font_cache[12] = ImageFont.load_default()

    NAMED_COLORS = {
        "black": (0, 0, 0, 255), "white": (255, 255, 255, 255),
        "red": (255, 0, 0, 255), "green": (0, 128, 0, 255),
        "blue": (0, 0, 255, 255), "yellow": (255, 255, 0, 255),
        "cyan": (0, 255, 255, 255), "magenta": (255, 0, 255, 255),
        "gray": (128, 128, 128, 255), "grey": (128, 128, 128, 255),
    }

    def _to_rgba(c):
        """Convert a reportlab color / CSS string to an (r,g,b,a) tuple."""
        if c is None:
            return None
        if isinstance(c, str):
            s = c.strip().lower()
            if s in ("none", "transparent"):
                return None
            if s in NAMED_COLORS:
                return NAMED_COLORS[s]
            if s.startswith("#"):
                h = s.lstrip("#")
                if len(h) == 3:
                    h = "".join(c2 * 2 for c2 in h)
                if len(h) == 6:
                    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
            if s.startswith("rgb("):
                nums = re.findall(r"[\d.]+", s)
                if len(nums) >= 3:
                    return (int(float(nums[0])), int(float(nums[1])), int(float(nums[2])), 255)
            return None
        if isinstance(c, tuple):
            if len(c) == 3:
                return (c[0], c[1], c[2], 255)
            if len(c) >= 4:
                return (c[0], c[1], c[2], int(c[3] * 255) if c[3] <= 1 else c[3])
        # reportlab Color object
        rgb = getattr(c, "rgb", None)
        if callable(rgb):
            rgb = rgb()
        if rgb is not None and len(rgb) >= 3:
            a = getattr(c, "alpha", 1.0) or 1.0
            return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), int(a * 255) if a <= 1 else int(a))
        return None

    def _font(size: int):
        size = max(6, int(round(size * scale)))
        if size not in font_cache:
            base = font_cache.get(12) or ImageFont.load_default()
            try:
                # Try scaling the base font
                font_cache[size] = base.font_variant(size=size) if hasattr(base, "font_variant") else base
            except Exception:
                font_cache[size] = base
        return font_cache[size]

    def _stroke_width(w):
        try:
            return max(1, int(round(float(w) * scale)))
        except (TypeError, ValueError):
            return 1

    def _transform_xy(x: float, y: float, m):
        """Apply reportlab-style affine transform matrix m=(a,b,c,d,e,f) to (x,y)."""
        if m is None:
            return x, y
        a, b, c, d, e, f = m
        return a * x + c * y + e, b * x + d * y + f

    def render(elem, parent_m=None, depth=0):
        # Combine parent transform with this element's transform
        own_m = getattr(elem, "transform", None)
        if own_m is not None and parent_m is not None:
            a1, b1, c1, d1, e1, f1 = parent_m
            a2, b2, c2, d2, e2, f2 = own_m
            m = (
                a1 * a2 + c1 * b2,
                b1 * a2 + d1 * b2,
                a1 * c2 + c1 * d2,
                b1 * c2 + d1 * d2,
                a1 * e2 + c1 * f2 + e1,
                b1 * e2 + d1 * f2 + f1,
            )
        elif own_m is not None:
            m = own_m
        else:
            m = parent_m

        if isinstance(elem, Rect):
            # reportlab Rect: x is left, y is BOTTOM-left (y-up)
            x0, y0 = _transform_xy(elem.x, elem.y, m)
            x1, y1 = _transform_xy(elem.x + elem.width, elem.y + elem.height, m)
            # Convert to top-down coords
            py0 = out_h - y1
            py1 = out_h - y0
            x0, x1 = sorted((x0 * scale, x1 * scale))
            py0, py1 = sorted((py0, py1))
            fill = _to_rgba(getattr(elem, "fillColor", None))
            stroke = _to_rgba(getattr(elem, "strokeColor", None))
            sw = _stroke_width(getattr(elem, "strokeWidth", 0) or 0)
            draw.rectangle([x0, py0, x1, py1],
                           fill=fill if fill else None,
                           outline=stroke if stroke and sw > 0 else None,
                           width=sw if stroke else 0)
        elif isinstance(elem, Line):
            x0, y0 = _transform_xy(elem.x1, elem.y1, m)
            x1, y1 = _transform_xy(elem.x2, elem.y2, m)
            stroke = _to_rgba(getattr(elem, "strokeColor", None))
            sw = _stroke_width(getattr(elem, "strokeWidth", 1) or 1)
            draw.line([x0 * scale, out_h - y0 * scale, x1 * scale, out_h - y1 * scale],
                      fill=stroke or (0, 0, 0, 255), width=sw)
        elif isinstance(elem, (Circle, Ellipse)):
            cx, cy = _transform_xy(elem.cx, elem.cy, m)
            rx = getattr(elem, "r", None) or getattr(elem, "rx", 0) or 0
            ry = getattr(elem, "r", None) or getattr(elem, "ry", 0) or 0
            fill = _to_rgba(getattr(elem, "fillColor", None))
            stroke = _to_rgba(getattr(elem, "strokeColor", None))
            sw = _stroke_width(getattr(elem, "strokeWidth", 0) or 0)
            bbox = [cx * scale - rx * scale, out_h - cy * scale - ry * scale,
                    cx * scale + rx * scale, out_h - cy * scale + ry * scale]
            draw.ellipse(bbox,
                         fill=fill if fill else None,
                         outline=stroke if stroke and sw > 0 else None,
                         width=sw if stroke else 0)
        elif isinstance(elem, Polygon):
            pts_ = getattr(elem, "points", [])
            if not pts_:
                pass
            else:
                pts = [_transform_xy(p.x, p.y, m) for p in pts_]
                screen = [(p[0] * scale, out_h - p[1] * scale) for p in pts]
                fill = _to_rgba(getattr(elem, "fillColor", None))
                stroke = _to_rgba(getattr(elem, "strokeColor", None))
                sw = _stroke_width(getattr(elem, "strokeWidth", 0) or 0)
                draw.polygon(screen,
                             fill=fill if fill else None,
                             outline=stroke if stroke and sw > 0 else None,
                             width=sw if stroke else 0)
        elif isinstance(elem, PolyLine):
            pts_ = getattr(elem, "points", [])
            if len(pts_) >= 2:
                pts = [_transform_xy(p.x, p.y, m) for p in pts_]
                screen = [(p[0] * scale, out_h - p[1] * scale) for p in pts]
                stroke = _to_rgba(getattr(elem, "strokeColor", None))
                sw = _stroke_width(getattr(elem, "strokeWidth", 1) or 1)
                draw.line(screen, fill=stroke or (0, 0, 0, 255),
                          width=sw, joint="curve")
        elif isinstance(elem, Path):
            # Parse path data and render as polylines / polygons
            d = getattr(elem, "d", "") or ""
            fill = _to_rgba(getattr(elem, "fillColor", None))
            stroke = _to_rgba(getattr(elem, "strokeColor", None))
            sw = _stroke_width(getattr(elem, "strokeWidth", 1) or 1)
            points, closed = _parse_path_to_points(d)
            if points:
                pts = [_transform_xy(x, y, m) for x, y in points]
                screen = [(p[0] * scale, out_h - p[1] * scale) for p in pts]
                if closed and fill:
                    draw.polygon(screen, fill=fill,
                                 outline=stroke if stroke and sw > 0 else None,
                                 width=sw if stroke else 0)
                else:
                    draw.line(screen, fill=stroke or (0, 0, 0, 255),
                              width=sw, joint="curve")
        elif isinstance(elem, String):
            x, y = _transform_xy(elem.x, elem.y, m)
            text = getattr(elem, "text", "") or ""
            font_size = getattr(elem, "fontSize", 12) or 12
            fill = _to_rgba(getattr(elem, "fillColor", None)) or (0, 0, 0, 255)
            draw.text((x * scale, out_h - y * scale), text,
                      fill=fill, font=_font(font_size))
        elif isinstance(elem, Group):
            for child in getattr(elem, "contents", []) or []:
                render(child, m, depth + 1)

    for child in drawing.contents or []:
        render(child, None)

    img.save(png_path, "PNG")


def _parse_path_to_points(d: str):
    """Parse a minimal subset of SVG path data and return (points, closed).

    Supports absolute (M/L/H/V/Z) and relative (m/l/h/v/z) commands.
    Sufficient for KiCad's schematic/PCB SVG output.
    """
    import re
    points: list[tuple[float, float]] = []
    if not d:
        return points, False
    tokens = re.findall(r"[MmLlHhVvZz]|-?\d+\.?\d*(?:[eE][-+]?\d+)?", d)
    cx, cy = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    closed = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
        else:
            # Implicit repeat of previous command — track it
            cmd = ""
        if cmd in ("M", "m"):
            if i + 1 >= len(tokens):
                break
            x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
            if cmd == "m" and points:
                x += cx; y += cy
            cx, cy = x, y
            start_x, start_y = cx, cy
            points.append((cx, cy))
        elif cmd in ("L", "l"):
            if i + 1 >= len(tokens):
                break
            x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
            if cmd == "l":
                x += cx; y += cy
            cx, cy = x, y
            points.append((cx, cy))
        elif cmd in ("H", "h"):
            if i >= len(tokens):
                break
            x = float(tokens[i]); i += 1
            if cmd == "h":
                x += cx
            cx = x
            points.append((cx, cy))
        elif cmd in ("V", "v"):
            if i >= len(tokens):
                break
            y = float(tokens[i]); i += 1
            if cmd == "v":
                y += cy
            cy = y
            points.append((cx, cy))
        elif cmd in ("Z", "z"):
            cx, cy = start_x, start_y
            points.append((cx, cy))
            closed = True
        else:
            # Unknown / unsupported command (curves etc.) — stop parsing
            break
    return points, closed


def _render_root_path() -> str:
    """Return the default output directory for rendered images."""
    return os.path.join(config.get_kcaa_data_dir(), "render")


def _crop_bbox_to_pixels(
    bbox_mm: tuple[float, float, float, float],
    extent_mm: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Convert an mm bbox within ``extent_mm`` to a pixel crop box.

    ``extent_mm`` is ``(x_min, y_min, x_max, y_max)``.  ``bbox_mm`` is
    ``(x, y, w, h)`` — top-left + size in mm, KiCad ``+Y down``.
    ``image_size`` is ``(width_px, height_px)``.

    The crop is clamped to the image bounds and rounded to integers.
    """
    ix_min, iy_min, ix_max, iy_max = extent_mm
    bx, by, bw, bh = bbox_mm
    src_w = max(ix_max - ix_min, 1e-9)
    src_h = max(iy_max - iy_min, 1e-9)
    px_w, px_h = image_size
    # x grows right, y grows down → both axes map monotonically.
    x0 = int(round((bx - ix_min) / src_w * px_w))
    y0 = int(round((by - iy_min) / src_h * px_h))
    x1 = int(round((bx + bw - ix_min) / src_w * px_w))
    y1 = int(round((by + bh - iy_min) / src_h * px_h))
    # Clamp to image bounds
    x0 = max(0, min(px_w, x0))
    x1 = max(0, min(px_w, x1))
    y0 = max(0, min(px_h, y0))
    y1 = max(0, min(px_h, y1))
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


def register_image_tools(mcp: FastMCP) -> None:
    """Register ``render_image`` with the MCP server."""

    @mcp.tool()
    async def render_image(
        kind: Literal["schematic", "pcb", "symbol", "footprint", "region"],
        schematic_path: str | None = None,
        pcb_path: str | None = None,
        lib_id: str | None = None,
        footprint_id: str | None = None,
        region_target: Literal["schematic", "pcb"] | None = None,
        bbox_x: float | None = None,
        bbox_y: float | None = None,
        bbox_width: float | None = None,
        bbox_height: float | None = None,
        output_format: Literal["png", "svg", "pdf"] = "png",
        output_path: str | None = None,
        width: int | None = None,
        height: int | None = None,
        layers: list[str] | None = None,
        background_color: str | None = None,
        ctx: Context | None = None,
    ) -> Image:
        """Render a KiCad schematic, PCB, symbol, footprint, or bbox region to an image.

        Returns MCP ``Image`` content (so the caller can pass the bytes
        directly to a vision-capable LLM) and also writes the image to
        ``output_path`` if provided, or to ``<kcaa_data_dir>/render/``
        otherwise.

        Args:
            kind: Which render target to use.

              - ``"schematic"`` — render the full page of ``schematic_path``.
              - ``"pcb"`` — render the full board of ``pcb_path``.
              - ``"symbol"`` — render the library symbol ``lib_id``
                (e.g. ``"Device:R_Small"``).
              - ``"footprint"`` — render the library footprint
                ``footprint_id`` (e.g. ``"Resistor_SMD:R_0805_2012Metric"``).
              - ``"region"`` — render a bbox-cropped region of a schematic
                or PCB.  Requires ``region_target`` plus ``bbox_x``,
                ``bbox_y``, ``bbox_width``, ``bbox_height``.  Output is
                always PNG (raster) regardless of ``output_format``.

            schematic_path: Required for ``kind="schematic"`` and as the
                source when ``kind="region"`` and ``region_target="schematic"``.
            pcb_path: Required for ``kind="pcb"`` and as the source when
                ``kind="region"`` and ``region_target="pcb"``.
            lib_id: Required for ``kind="symbol"``.
            footprint_id: Required for ``kind="footprint"``.
            region_target: ``"schematic"`` or ``"pcb"``, required for ``kind="region"``.
            bbox_x, bbox_y: Top-left of the region in mm (KiCad ``+Y down``).
            bbox_width, bbox_height: Region extent in mm.
            output_format: ``"png"`` (default, vision-friendly), ``"svg"``,
                or ``"pdf"``.  Forced to ``"png"`` for ``kind="region"``.
            output_path: Where to write the image.  Defaults to
                ``<kcaa_data_dir>/render/<name>.<ext>``.
            width, height: Render size hints in pixels (forwarded to
                ``kicad-cli``).  ``height`` is auto-computed from aspect
                if omitted.  Default: ``1600``.
            layers: PCB layers to include (kicad-cli ``--layers`` syntax,
                comma-separated).  Defaults to
                ``F.Cu,B.Cu,F.SilkS,B.SilkS,F.Mask,B.Mask,Edge.Cuts``.
                Only meaningful for ``kind="pcb"`` and PCB regions.
            background_color: Optional hex color (e.g. ``"#FFFFFF"``) for
                the page background.  Forwarded to ``kicad-cli``.
            ctx: MCP context (used for progress + logging).
        """

        async def _info(msg: str) -> None:
            if ctx:
                await ctx.info(msg)
            log.info(msg)

        # --- validate kind/source ---
        if kind in ("schematic", "region") and (
            kind == "schematic"
            and not schematic_path
            or kind == "region"
            and region_target == "schematic"
            and not schematic_path
        ):
            raise ValueError(f"kind={kind!r} requires schematic_path")
        if kind == "pcb" and not pcb_path:
            raise ValueError("kind='pcb' requires pcb_path")
        if kind == "region" and not region_target:
            raise ValueError("kind='region' requires region_target")
        if kind == "region":
            if region_target == "pcb" and not pcb_path:
                raise ValueError("region_target='pcb' requires pcb_path")
            if (
                bbox_x is None
                or bbox_y is None
                or bbox_width is None
                or bbox_height is None
            ):
                raise ValueError(
                    "kind='region' requires bbox_x, bbox_y, bbox_width, bbox_height"
                )
        if kind == "symbol" and not lib_id:
            raise ValueError("kind='symbol' requires lib_id")
        if kind == "footprint" and not footprint_id:
            raise ValueError("kind='footprint' requires footprint_id")

        # kicad-cli must be on PATH / configured for any of these modes.
        try:
            get_kicad_cli_path(required=True)
        except KiCadCLIError as e:
            raise RuntimeError(str(e)) from e

        # --- region mode forces raster ---
        effective_format: Literal["png", "svg", "pdf"] = "png" if kind == "region" else output_format

        # --- determine render width (used for SVG→PNG conversion only) ---
        render_width = width if width is not None else 1600

        # --- build kicad-cli args (always SVG — kicad-cli export doesn't do PNG) ---
        cli_args: list[str]
        extent_mm: tuple[float, float, float, float] | None = None  # for region
        layers_str = ",".join(layers) if layers else _DEFAULT_PCB_LAYERS

        if kind == "schematic":
            assert schematic_path is not None
            cli_args = ["sch", "export", "svg"]
            if background_color:
                cli_args += ["--background-color", background_color]
            cli_args += [schematic_path]

        elif kind == "pcb":
            assert pcb_path is not None
            cli_args = [
                "pcb",
                "export",
                "svg",
                "--layers",
                layers_str,
            ]
            if background_color:
                cli_args += ["--background-color", background_color]
            cli_args += [pcb_path]

        elif kind == "symbol":
            assert lib_id is not None
            cli_args = [
                "sym",
                "export",
                "svg",
                "--symbol",
                lib_id,
            ]
            if background_color:
                cli_args += ["--background-color", background_color]

        elif kind == "footprint":
            assert footprint_id is not None
            cli_args = [
                "fp",
                "export",
                "svg",
                "--footprint",
                footprint_id,
            ]
            if background_color:
                cli_args += ["--background-color", background_color]

        elif kind == "region":
            assert region_target is not None
            await _info(f"render_image: region mode, target={region_target}")
            if region_target == "schematic":
                assert schematic_path is not None
                extent_mm = (0.0, 0.0, *_schematic_page_mm(schematic_path))
                cli_args = ["sch", "export", "svg"]
                if background_color:
                    cli_args += ["--background-color", background_color]
                cli_args += [schematic_path]
            else:  # region_target == "pcb"
                assert pcb_path is not None
                extent_mm = _pcb_bbox_mm(pcb_path)
                cli_args = [
                    "pcb",
                    "export",
                    "svg",
                    "--layers",
                    layers_str,
                ]
                if background_color:
                    cli_args += ["--background-color", background_color]
                cli_args += [pcb_path]

        else:  # pragma: no cover — defensive
            raise ValueError(f"unknown kind: {kind!r}")

        # --- choose output path (always .svg for the kicad-cli output) ---
        if output_path is None:
            os.makedirs(_render_root_path(), exist_ok=True)
            stem = _default_stem(kind, schematic_path, pcb_path, lib_id, footprint_id)
            svg_path = os.path.join(_render_root_path(), stem + ".svg")
        else:
            out_dir = os.path.dirname(output_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            # kicad-cli writes whatever extension matches the format; pin to .svg
            base, _ = os.path.splitext(output_path)
            svg_path = base + ".svg"

        await _info(f"render_image: kind={kind} → SVG via kicad-cli at {svg_path}")

        # --- run kicad-cli to produce SVG ---
        try:
            proc = await _run_cli(cli_args, svg_path)
        except Exception as e:
            raise RuntimeError(f"kicad-cli invocation failed for {kind}: {e}") from e

        if proc.returncode != 0:
            raise RuntimeError(
                f"kicad-cli failed (exit={proc.returncode}) for kind={kind}: "
                f"{(proc.stderr or proc.stdout or '').strip()}"
            )

        # --- load SVG bytes (always present) ---
        with open(svg_path, "rb") as f:
            svg_bytes = f.read()

        # --- convert SVG → PNG if needed (default + region) ---
        if effective_format == "png":
            png_path = os.path.splitext(svg_path)[0] + ".png"
            try:
                _svg_to_png(
                    svg_path,
                    png_path,
                    output_width=render_width,
                    background_color=background_color,
                )
            except Exception as e:
                raise RuntimeError(f"SVG→PNG conversion failed: {e}") from e
            data, pil = _load_image(png_path, "png")

            # Region mode: Pillow-crop the rasterised PNG.
            if kind == "region" and pil is not None and extent_mm is not None:
                assert (
                    bbox_x is not None
                    and bbox_y is not None
                    and bbox_width is not None
                    and bbox_height is not None
                )
                crop_box = _crop_bbox_to_pixels(
                    (bbox_x, bbox_y, bbox_width, bbox_height),
                    extent_mm,
                    pil.size,
                )
                cropped = pil.crop(crop_box)
                cropped.save(png_path, format="PNG")
                data, pil = _load_image(png_path, "png")
            final_path = png_path
        else:
            # SVG or PDF — return raw bytes from the kicad-cli output.
            data, pil = svg_bytes, None
            final_path = svg_path

        await _info(
            f"render_image: wrote {final_path} ({len(data)} bytes"
            + (f", {pil.size[0]}x{pil.size[1]} px" if pil else "")
            + ")"
        )

        return Image(data=data, format=effective_format)


def _default_stem(
    kind: str,
    schematic_path: str | None,
    pcb_path: str | None,
    lib_id: str | None,
    footprint_id: str | None,
) -> str:
    """Pick a sensible default filename stem for a render."""
    base: str
    if kind == "schematic" and schematic_path:
        base = os.path.splitext(os.path.basename(schematic_path))[0]
    elif kind == "pcb" and pcb_path:
        base = os.path.splitext(os.path.basename(pcb_path))[0]
    elif kind == "symbol" and lib_id:
        base = "sym_" + lib_id.replace(":", "_")
    elif kind == "footprint" and footprint_id:
        base = "fp_" + footprint_id.replace(":", "_")
    elif kind == "region":
        base = "region"
    else:
        base = "render"
    return base
