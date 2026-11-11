"""Coordinate helpers. pyproj is imported only when a projection is actually needed."""
from __future__ import annotations

import functools

import numpy as np

UTM = "EPSG:32632"     # the friction surface's grid
WGS84 = "EPSG:4326"    # facilities, boundaries, population


@functools.lru_cache(maxsize=None)
def _transformer(src: str, dst: str):
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True)


def to_utm(lon, lat):
    x, y = _transformer(WGS84, UTM).transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def to_lonlat(x, y):
    lon, lat = _transformer(UTM, WGS84).transform(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(lon), np.asarray(lat)


def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km. Arguments broadcast."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(a, float)) for a in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def mercator(lon, lat):
    """Longitude and latitude to Web Mercator metres (EPSG:3857), without pyproj."""
    r = 6378137.0
    lon = np.radians(np.asarray(lon, float))
    lat = np.radians(np.clip(np.asarray(lat, float), -85.05, 85.05))
    return r * lon, r * np.log(np.tan(np.pi / 4 + lat / 2))


def rowcol(transform, x, y):
    """Row and column of the cells holding points (x, y), for a north-up affine transform."""
    a, _, c, _, e, f = transform[:6]
    col = np.floor((np.asarray(x, float) - c) / a).astype(np.int64)
    row = np.floor((np.asarray(y, float) - f) / e).astype(np.int64)
    return row, col


def cell_centres(transform, rows, cols):
    a, _, c, _, e, f = transform[:6]
    return c + (np.asarray(cols) + 0.5) * a, f + (np.asarray(rows) + 0.5) * e
