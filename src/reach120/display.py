"""Map images and colours shared by the app and the brief.

Travel-time bands are ordered, so they take one hue, light to dark: a blue sequential ramp
(steps 250, 400, 550, 700 of a reference palette). Its light end clears 2:1
on the light map surface, as an ordinal ramp must. "No route found" is a neutral grey, off
the ramp, because it is not a distance.

Map markers sit on that blue ramp, so the one highlighted series (recommended upgrades)
takes the next categorical hue, orange, as a second context would. Existing hospitals are
reference marks in primary ink with a white ring.
"""
from __future__ import annotations

import numpy as np

# (label, upper bound in minutes, colour)
BANDS = [
    ("Within 30 minutes", 30, "#86b6ef"),
    ("30 to 60 minutes", 60, "#3987e5"),
    ("1 to 2 hours", 120, "#1c5cab"),
    ("Over 2 hours", np.inf, "#0d366b"),
]
NO_ROUTE = ("No route found", "#898781")

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
UPGRADE = "#eb6834"          # categorical slot 2 (see module docstring)

# Lines in the budget chart: categorical slots 1-3, light and dark steps. Validated with
# the palette checker: all pass; light aqua is below 3:1, so the chart labels its lines directly.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70"]

# Ward choropleth (share of women more than two hours away): the same blue ramp, sequential.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def classify(minutes: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """0 outside the state, 1..4 the bands above, 5 no route."""
    out = np.zeros(minutes.shape, dtype=np.uint8)
    finite = np.isfinite(minutes)
    lower = -np.inf
    for i, (_, upper, _) in enumerate(BANDS, start=1):
        out[inside & finite & (minutes > lower) & (minutes <= upper)] = i
        lower = upper
    out[inside & ~finite] = 5
    return out


def class_colours(alpha: int = 225) -> np.ndarray:
    lut = np.zeros((6, 4), dtype=np.uint8)
    for i, (_, _, colour) in enumerate(BANDS, start=1):
        lut[i] = (*hex_rgb(colour), alpha)
    lut[5] = (*hex_rgb(NO_ROUTE[1]), alpha)
    return lut


def write_png(classes: np.ndarray, src_transform, src_crs: str, path) -> list[float]:
    """Reproject a class raster to Web Mercator, colour it, save a PNG.

    Returns [west, south, east, north] in degrees, the corners web maps need.
    """
    from affine import Affine
    from PIL import Image
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    h, w = classes.shape
    a, _, c, _, e, f = src_transform[:6]
    bounds = (c, f + e * h, c + a * w, f)
    dst_t, dw, dh = calculate_default_transform(src_crs, "EPSG:3857", w, h, *bounds)
    dst = np.zeros((dh, dw), dtype=np.uint8)
    reproject(source=classes, destination=dst, src_transform=Affine(*src_transform[:6]), src_crs=src_crs,
              src_nodata=0, dst_transform=dst_t, dst_crs="EPSG:3857", dst_nodata=0, resampling=Resampling.nearest)
    rgba = class_colours()[dst]
    Image.fromarray(rgba).save(path, optimize=True)
    r = 6378137.0
    west = np.degrees(dst_t.c / r)
    east = np.degrees((dst_t.c + dst_t.a * dw) / r)
    north = np.degrees(2 * np.arctan(np.exp(dst_t.f / r)) - np.pi / 2)
    south = np.degrees(2 * np.arctan(np.exp((dst_t.f + dst_t.e * dh) / r)) - np.pi / 2)
    return [float(west), float(south), float(east), float(north)]


def share_colour(share) -> list[int]:
    """Fill for a ward's share of women more than two hours away (0-1)."""
    if share is None or not np.isfinite(share):
        return [*hex_rgb(MUTED), 90]
    idx = min(int(share * len(SEQUENTIAL)), len(SEQUENTIAL) - 1)
    return [*hex_rgb(SEQUENTIAL[idx]), 200]


def spread_labels(values: dict, min_gap: float) -> dict:
    """Vertical label positions near `values`, pushed apart so no two are closer than `min_gap`."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    out, last = {}, None
    for name, v in items:
        pos = v if last is None else max(v, last + min_gap)
        out[name] = pos
        last = pos
    return out


def format_naira(value: float) -> str:
    v = float(value)
    if abs(v) >= 1e9:
        return f"₦{v / 1e9:,.2f} billion"
    if abs(v) >= 1e6:
        return f"₦{v / 1e6:,.0f} million"
    return f"₦{v:,.0f}"
