"""Read a window from a tiled, uncompressed GeoTIFF stored inside a zip, without unzipping it.

Each GRID3 friction surface is one 6.39 GB GeoTIFF inside a deflate-compressed zip. This
laptop cannot hold it unzipped, and a deflate stream cannot be read at random. So:

1. ``read_layout`` streams through the member once to find the tag directory (GDAL and
   ArcGIS often write it at the end) and records each tile's byte offset. The result is
   cached, so this costs one pass per surface, ever.
2. ``read_window`` streams forward once more, keeping only the tiles the window needs,
   and stops after the last of them.

Only what the friction surfaces need is supported: classic or BigTIFF, one band,
uncompressed, tiled.
"""
from __future__ import annotations

import json
import math
import struct
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_TYPES = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 6: ("b", 1), 7: ("B", 1),
          8: ("h", 2), 9: ("i", 4), 10: ("ii", 8), 11: ("f", 4), 12: ("d", 8), 16: ("Q", 8),
          17: ("q", 8), 18: ("Q", 8)}
_CHUNK = 64 << 20
_KEEP_BEFORE = 96 << 20      # bytes kept before the tag directory, where long arrays usually sit
_KEEP_AFTER = 96 << 20


@dataclass
class TiffLayout:
    width: int
    height: int
    tile_width: int
    tile_height: int
    dtype: str                 # numpy dtype string, with byte order
    nodata: float | None
    pixel_width: float
    pixel_height: float
    origin_x: float            # outer corner of the top-left pixel
    origin_y: float
    epsg: int | None
    member_size: int
    offsets: np.ndarray
    counts: np.ndarray

    @property
    def tiles_across(self) -> int:
        return math.ceil(self.width / self.tile_width)

    @property
    def tiles_down(self) -> int:
        return math.ceil(self.height / self.tile_height)

    @property
    def transform(self) -> tuple:
        """Affine coefficients (a, b, c, d, e, f) for the whole raster."""
        return (self.pixel_width, 0.0, self.origin_x, 0.0, -self.pixel_height, self.origin_y)


class _Buffers:
    def __init__(self):
        self.parts: list[tuple[int, bytes]] = []

    def add(self, start: int, data: bytes) -> None:
        self.parts.append((start, data))

    def get(self, offset: int, length: int):
        for start, data in self.parts:
            if start <= offset and offset + length <= start + len(data):
                return data[offset - start: offset - start + length]
        return None


def _progress(label: str, done: int, total: int, t0: float) -> None:
    rate = done / max(time.time() - t0, 1e-6) / 1e6
    print(f"    {label}: {done / 1e9:5.2f} of {total / 1e9:.2f} GB ({rate:.0f} MB/s)", file=sys.stderr, flush=True)


def _skip(f, n: int, label: str, pos: int, total: int, t0: float, last_report: list) -> int:
    while n > 0:
        got = len(f.read(min(n, _CHUNK)))
        if got == 0:
            raise EOFError("zip member ended early")
        n -= got
        pos += got
        if pos - last_report[0] >= (1 << 30):
            _progress(label, pos, total, t0)
            last_report[0] = pos
    return pos


def _read_regions(zip_path: Path, member: str, regions, label: str) -> _Buffers:
    """Read byte ranges of a zip member in one forward pass."""
    merged: list[list[int]] = []
    for start, length in sorted((int(s), int(n)) for s, n in regions if n > 0):
        end = start + length
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    bufs = _Buffers()
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(member)
        total = info.file_size
        with zf.open(info) as f:
            pos, t0, last = 0, time.time(), [0]
            for start, end in merged:
                end = min(end, total)
                pos = _skip(f, start - pos, label, pos, total, t0, last)
                data = f.read(end - start)
                pos += len(data)
                bufs.add(start, data)
    return bufs


def _header(zip_path: Path, member: str):
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(member)
        with zf.open(info) as f:
            head = f.read(16)
    order = head[:2]
    if order not in (b"II", b"MM"):
        raise ValueError(f"{member} is not a TIFF")
    bo = "<" if order == b"II" else ">"
    magic = struct.unpack(bo + "H", head[2:4])[0]
    if magic == 43:
        big = True
        ifd = struct.unpack(bo + "Q", head[8:16])[0]
    elif magic == 42:
        big = False
        ifd = struct.unpack(bo + "I", head[4:8])[0]
    else:
        raise ValueError(f"{member}: unknown TIFF magic number {magic}")
    return bo, big, ifd, info.file_size


def _parse_entries(bufs: _Buffers, bo: str, big: bool, ifd: int):
    count_fmt, count_size, entry_size, off_fmt = ("Q", 8, 20, "Q") if big else ("H", 2, 12, "I")
    raw = bufs.get(ifd, count_size)
    if raw is None:
        raise ValueError("TIFF tag directory not found in the bytes read")
    n = struct.unpack(bo + count_fmt, raw)[0]
    block = bufs.get(ifd + count_size, n * entry_size)
    if block is None:
        raise ValueError("TIFF tag directory truncated")
    entries = {}
    for i in range(n):
        e = block[i * entry_size:(i + 1) * entry_size]
        if big:
            tag, typ, cnt = struct.unpack(bo + "HHQ", e[:12])
            raw_val = e[12:20]
        else:
            tag, typ, cnt = struct.unpack(bo + "HHI", e[:8])
            raw_val = e[8:12]
        entries[tag] = (typ, cnt, raw_val)
    inline_size = 8 if big else 4
    return entries, inline_size, off_fmt


def _value_location(entry, bo, inline_size, off_fmt):
    """(inline bytes or None, offset, byte length) for a tag value."""
    typ, cnt, raw_val = entry
    if typ not in _TYPES:
        return None, None, 0
    size = _TYPES[typ][1] * cnt
    if size <= inline_size:
        return raw_val[:size], None, size
    return None, struct.unpack(bo + off_fmt, raw_val)[0], size


def _decode(entry, data: bytes, bo: str):
    typ, cnt, _ = entry
    fmt = _TYPES[typ][0]
    if typ == 2:
        return data.rstrip(b"\x00").decode("latin1")
    if typ in (5, 10):
        vals = struct.unpack(bo + fmt[0] * (2 * cnt), data)
        return tuple(vals[i] / vals[i + 1] for i in range(0, len(vals), 2))
    if typ in (16, 18):
        return np.frombuffer(data, dtype=bo + "u8")
    if typ == 4 and cnt > 64:
        return np.frombuffer(data, dtype=bo + "u4")
    return struct.unpack(bo + fmt * cnt, data)


def _cache_path(cache_dir: Path, zip_path: Path, member: str) -> Path:
    st = zip_path.stat()
    key = f"{zip_path.name}.{Path(member).name}.{st.st_size}.{int(st.st_mtime)}"
    return Path(cache_dir) / f"{key}.layout.npz"


def read_layout(zip_path, member: str, cache_dir=None) -> TiffLayout:
    """Find the tag directory and tile offsets of a TIFF inside a zip. Cached."""
    zip_path = Path(zip_path)
    cpath = _cache_path(cache_dir, zip_path, member) if cache_dir else None
    if cpath and cpath.exists():
        with np.load(cpath, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return TiffLayout(offsets=z["offsets"], counts=z["counts"], **meta)

    bo, big, ifd, size = _header(zip_path, member)
    regions = [(0, min(size, 1 << 20)), (max(0, ifd - _KEEP_BEFORE), min(size, ifd + _KEEP_AFTER) - max(0, ifd - _KEEP_BEFORE))]
    print(f"  reading the tag directory of {Path(member).name} (one pass, cached afterwards)", file=sys.stderr, flush=True)
    bufs = _read_regions(zip_path, member, regions, "tag directory")
    entries, inline_size, off_fmt = _parse_entries(bufs, bo, big, ifd)

    wanted = [256, 257, 258, 259, 277, 284, 322, 323, 324, 325, 339, 33550, 33922, 34735, 42113]
    values, missing = {}, []
    for tag in wanted:
        if tag not in entries:
            continue
        inline, offset, length = _value_location(entries[tag], bo, inline_size, off_fmt)
        if inline is not None:
            values[tag] = _decode(entries[tag], inline, bo)
            continue
        data = bufs.get(offset, length)
        if data is None:
            missing.append((tag, offset, length))
        else:
            values[tag] = _decode(entries[tag], data, bo)
    if missing:
        more = _read_regions(zip_path, member, [(o, n) for _, o, n in missing], "tag values")
        for tag, offset, length in missing:
            values[tag] = _decode(entries[tag], more.get(offset, length), bo)

    def first(tag, default=None):
        v = values.get(tag)
        if v is None:
            return default
        return v[0] if isinstance(v, (tuple, list, np.ndarray)) else v

    if 322 not in values:
        raise NotImplementedError("only tiled TIFFs are supported")
    if first(259, 1) != 1:
        raise NotImplementedError(f"compression {first(259)} is not supported; the tiles must be uncompressed")
    if first(277, 1) != 1:
        raise NotImplementedError("only single-band rasters are supported")
    bits, fmt = first(258, 8), first(339, 1)
    kind = {1: "u", 2: "i", 3: "f"}[fmt]
    dtype = np.dtype(f"{bo}{kind}{bits // 8}").str

    scale = values.get(33550)
    tie = values.get(33922)
    if scale is None or tie is None:
        raise ValueError("the TIFF has no georeferencing tags")
    pw, ph = float(scale[0]), float(scale[1])
    origin_x = float(tie[3]) - float(tie[0]) * pw
    origin_y = float(tie[4]) + float(tie[1]) * ph
    epsg, raster_type = None, 1
    gk = values.get(34735)
    if gk is not None:
        gk = list(gk)
        for i in range(4, 4 + 4 * int(gk[3]), 4):
            key_id, location, _, value = gk[i:i + 4]
            if location == 0 and key_id == 3072:
                epsg = int(value)
            elif location == 0 and key_id == 2048 and epsg is None:
                epsg = int(value)
            elif location == 0 and key_id == 1025:
                raster_type = int(value)
    if raster_type == 2:          # PixelIsPoint: tie point is a pixel centre
        origin_x -= pw / 2
        origin_y += ph / 2
    nodata = values.get(42113)
    nodata = float(nodata) if isinstance(nodata, str) and nodata.strip() else None

    layout = TiffLayout(
        width=int(first(256)), height=int(first(257)), tile_width=int(first(322)),
        tile_height=int(first(323)), dtype=dtype, nodata=nodata, pixel_width=pw, pixel_height=ph,
        origin_x=origin_x, origin_y=origin_y, epsg=epsg, member_size=int(size),
        offsets=np.asarray(values[324], dtype=np.uint64), counts=np.asarray(values[325], dtype=np.uint64))
    if layout.offsets.size != layout.tiles_across * layout.tiles_down:
        raise ValueError("tile offsets do not match the tile grid")
    if cpath:
        meta = {k: v for k, v in asdict(layout).items() if k not in ("offsets", "counts")}
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cpath, offsets=layout.offsets, counts=layout.counts, meta=json.dumps(meta))
    return layout


def find_member(zip_path, name: str) -> str:
    """The member whose file name is `name` (folders inside the zip differ between surfaces)."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if not info.is_dir() and Path(info.filename).name == name:
                return info.filename
    raise FileNotFoundError(f"{name} not found inside {zip_path}")


def read_window(zip_path, member: str, layout: TiffLayout, col0: int, row0: int, ncols: int, nrows: int) -> np.ndarray:
    """Pixels [row0:row0+nrows, col0:col0+ncols] of the raster, in one forward pass."""
    if col0 < 0 or row0 < 0 or col0 + ncols > layout.width or row0 + nrows > layout.height:
        raise ValueError("window extends beyond the raster")
    dtype = np.dtype(layout.dtype)
    fill = layout.nodata if layout.nodata is not None else 0
    out = np.full((nrows, ncols), fill, dtype=dtype.newbyteorder("="))
    tw, th = layout.tile_width, layout.tile_height
    tile_bytes = tw * th * dtype.itemsize
    jobs = []
    for tr in range(row0 // th, (row0 + nrows - 1) // th + 1):
        for tc in range(col0 // tw, (col0 + ncols - 1) // tw + 1):
            idx = tr * layout.tiles_across + tc
            off, cnt = int(layout.offsets[idx]), int(layout.counts[idx])
            if cnt == 0:
                continue      # sparse tile: stays nodata
            if cnt != tile_bytes:
                raise ValueError(f"tile {idx} has {cnt} bytes, expected {tile_bytes} (compressed tiles are not supported)")
            jobs.append((off, tr, tc))
    jobs.sort()
    t0, last = time.time(), [0]
    with zipfile.ZipFile(zip_path) as zf, zf.open(zf.getinfo(member)) as f:
        pos = 0
        for off, tr, tc in jobs:
            if off < pos:
                raise ValueError("overlapping tiles")
            pos = _skip(f, off - pos, "window", pos, layout.member_size, t0, last)
            buf = f.read(tile_bytes)
            pos += len(buf)
            tile = np.frombuffer(buf, dtype=dtype).reshape(th, tw)
            r_start, c_start = tr * th, tc * tw
            rr0, rr1 = max(r_start, row0), min(r_start + th, row0 + nrows)
            cc0, cc1 = max(c_start, col0), min(c_start + tw, col0 + ncols)
            out[rr0 - row0:rr1 - row0, cc0 - col0:cc1 - col0] = tile[rr0 - r_start:rr1 - r_start, cc0 - c_start:cc1 - c_start]
    return out
