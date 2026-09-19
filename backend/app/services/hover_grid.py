"""
hover_grid.py — compact per-cell grid for the map hover readout.

The frontend used to power its hover tooltip by downloading the full per-cell
prediction CSV: 5.6 MB of text, 1.8 MB on the wire, about four seconds after
every prediction, parsed into 68,276 JavaScript objects and then searched
linearly on every mouse move.

This writes the same information as two raster-shaped arrays instead. The map
overlay is a PNG of exactly these dimensions stretched onto a latitude and
longitude box, so the browser can turn a cursor position into an array index
arithmetically and read the answer directly. Measured on a real scenario the
file is 178 KB gzipped, a tenth of the CSV, and the lookup stops being a search.

Layout, little-endian throughout:

    offset  bytes        field
    0       4            magic "MBHG"
    4       1            format version
    5       1            reserved
    6       2            depth scale (units per metre; 100 means centimetres)
    8       4            rows
    12      4            cols
    16      2            depth nodata value
    18      1            hazard nodata value
    19      5            reserved
    24      rows*cols    hazard class code, uint8
    ...     rows*cols*2  depth, uint16 in 1/scale metre
"""

from pathlib import Path

import numpy as np

MAGIC = b"MBHG"
VERSION = 1
DEPTH_SCALE = 100          # store depth in centimetres
DEPTH_NODATA = 65535
HEADER_BYTES = 24

# uint16 centimetres tops out at 655.35 m, far above any predicted depth, but
# clamp anyway so a pathological value wraps to the ceiling rather than to zero.
_MAX_DEPTH_M = (DEPTH_NODATA - 1) / DEPTH_SCALE


def save_hover_grid(
    path: Path,
    depth_raster: np.ndarray,
    hazard_raster: np.ndarray,
    hazard_nodata: int,
    depth_nodata_in: float,
) -> Path:
    """Write the packed grid for one scenario and return its path."""
    rows, cols = hazard_raster.shape
    if depth_raster.shape != hazard_raster.shape:
        raise ValueError(
            f"depth raster {depth_raster.shape} and hazard raster "
            f"{hazard_raster.shape} must have the same shape."
        )

    valid = depth_raster > (depth_nodata_in + 1.0)      # nodata is a large negative
    scaled = np.where(
        valid,
        np.clip(depth_raster, 0.0, _MAX_DEPTH_M) * DEPTH_SCALE,
        DEPTH_NODATA,
    ).astype("<u2")

    header = bytearray(HEADER_BYTES)
    header[0:4] = MAGIC
    header[4] = VERSION
    header[6:8] = int(DEPTH_SCALE).to_bytes(2, "little")
    header[8:12] = int(rows).to_bytes(4, "little")
    header[12:16] = int(cols).to_bytes(4, "little")
    header[16:18] = int(DEPTH_NODATA).to_bytes(2, "little")
    header[18] = int(hazard_nodata) & 0xFF

    hazard_bytes = np.ascontiguousarray(hazard_raster, dtype=np.uint8).tobytes()

    # Keep the uint16 block on an even offset so the browser can wrap it in a
    # Uint16Array view directly instead of copying the buffer.
    pad = b"\x00" * (len(hazard_bytes) % 2)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(hazard_bytes)
        f.write(pad)
        f.write(scaled.tobytes())
    return path
