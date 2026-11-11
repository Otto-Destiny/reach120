"""The zip-streamed TIFF reader, on synthetic files with the friction surface's structure."""
import struct
import zipfile

import numpy as np
import pytest

from reach120 import tiffzip


def _inline(fmt: str, values, size: int) -> bytes:
    raw = struct.pack("<" + fmt * len(values), *values)
    return raw.ljust(size, b"\0")


def write_tiff(path, arr: np.ndarray, tile: int, big: bool, origin=(100000.0, 900000.0), px=30.0, nodata=-9999.0):
    """A tiled, uncompressed float32 GeoTIFF with tiles in shuffled order and the tag directory at the end."""
    h, w = arr.shape
    across, down = -(-w // tile), -(-h // tile)
    tiles = []
    for tr in range(down):
        for tc in range(across):
            t = np.full((tile, tile), nodata, np.float32)
            block = arr[tr * tile:(tr + 1) * tile, tc * tile:(tc + 1) * tile]
            t[:block.shape[0], :block.shape[1]] = block
            tiles.append(t.astype("<f4").tobytes())
    order = np.random.default_rng(3).permutation(len(tiles))
    data = bytearray(16 if big else 8)
    offsets = [0] * len(tiles)
    for idx in order:
        offsets[idx] = len(data)
        data += tiles[idx]

    def put(raw: bytes) -> int:
        off = len(data)
        data.extend(raw)
        if len(data) % 2:
            data.append(0)
        return off

    long_fmt, long_type = ("Q", 16) if big else ("I", 4)
    off_tiles = put(struct.pack("<" + long_fmt * len(offsets), *offsets))
    off_counts = put(struct.pack("<" + long_fmt * len(tiles), *[len(tiles[0])] * len(tiles)))
    off_scale = put(struct.pack("<3d", px, px, 0.0))
    off_tie = put(struct.pack("<6d", 0, 0, 0, origin[0], origin[1], 0))
    geokeys = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32632]
    off_geo = put(struct.pack("<16H", *geokeys))
    nd = (repr(nodata) + "\0").encode()
    inline = 8 if big else 4
    off_nd = put(nd) if len(nd) > inline else None

    def ref(off):
        return struct.pack("<" + long_fmt, off)

    entries = [
        (256, 4, 1, _inline("I", [w], inline)), (257, 4, 1, _inline("I", [h], inline)),
        (258, 3, 1, _inline("H", [32], inline)), (259, 3, 1, _inline("H", [1], inline)),
        (262, 3, 1, _inline("H", [1], inline)), (277, 3, 1, _inline("H", [1], inline)),
        (284, 3, 1, _inline("H", [1], inline)), (322, 3, 1, _inline("H", [tile], inline)),
        (323, 3, 1, _inline("H", [tile], inline)), (324, long_type, len(tiles), ref(off_tiles)),
        (325, long_type, len(tiles), ref(off_counts)), (339, 3, 1, _inline("H", [3], inline)),
        (33550, 12, 3, ref(off_scale)), (33922, 12, 6, ref(off_tie)), (34735, 3, 16, ref(off_geo)),
        (42113, 2, len(nd), nd.ljust(inline, b"\0") if off_nd is None else ref(off_nd)),
    ]
    ifd = len(data)
    if big:
        data += struct.pack("<Q", len(entries))
        for tag, typ, cnt, val in entries:
            data += struct.pack("<HHQ", tag, typ, cnt) + val
        data += struct.pack("<Q", 0)
        data[:16] = b"II" + struct.pack("<HHHQ", 43, 8, 0, ifd)
    else:
        data += struct.pack("<H", len(entries))
        for tag, typ, cnt, val in entries:
            data += struct.pack("<HHI", tag, typ, cnt) + val
        data += struct.pack("<I", 0)
        data[:8] = b"II" + struct.pack("<HI", 42, ifd)
    path.write_bytes(bytes(data))


@pytest.mark.parametrize("big", [True, False])
def test_window_matches_source(tmp_path, big):
    rng = np.random.default_rng(0)
    arr = rng.uniform(0.001, 0.05, size=(203, 317)).astype(np.float32)
    tif = tmp_path / "surface.tif"
    write_tiff(tif, arr, tile=32, big=big)
    zpath = tmp_path / "surface.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tif, arcname="inner/surface.tif")

    member = tiffzip.find_member(zpath, "surface.tif")
    lay = tiffzip.read_layout(zpath, member, cache_dir=tmp_path / "cache")
    assert (lay.width, lay.height, lay.tile_width, lay.epsg, lay.nodata) == (317, 203, 32, 32632, -9999.0)
    assert lay.transform == (30.0, 0.0, 100000.0, 0.0, -30.0, 900000.0)

    for col0, row0, ncols, nrows in [(0, 0, 317, 203), (5, 7, 100, 60), (300, 190, 17, 13), (31, 31, 2, 2)]:
        got = tiffzip.read_window(zpath, member, lay, col0, row0, ncols, nrows)
        np.testing.assert_array_equal(got, arr[row0:row0 + nrows, col0:col0 + ncols])

    cached = tiffzip.read_layout(zpath, member, cache_dir=tmp_path / "cache")
    np.testing.assert_array_equal(cached.offsets, lay.offsets)
    assert cached.dtype == lay.dtype


def test_window_outside_raster_is_rejected(tmp_path):
    arr = np.ones((40, 40), np.float32)
    tif = tmp_path / "s.tif"
    write_tiff(tif, arr, tile=16, big=True)
    zpath = tmp_path / "s.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tif, arcname="s.tif")
    lay = tiffzip.read_layout(zpath, "s.tif")
    with pytest.raises(ValueError):
        tiffzip.read_window(zpath, "s.tif", lay, 30, 30, 20, 5)
