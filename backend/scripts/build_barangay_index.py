"""
build_barangay_index.py — precompute the grid-cell -> barangay assignment.

Run this once whenever the barangay boundaries or the spatial grid change:

    python scripts/build_barangay_index.py

It writes two files into assets/tables/:

  barangay_meta.json   human-readable: barangay names, per-barangay cell totals,
                       the Sipocot-wide cell total used as the legend denominator,
                       and the channel-cell definition.
  barangay_index.npz   compact arrays: the barangay index of every ACTIVE grid
                       cell (row-aligned with grid_cells.csv) and a boolean
                       channel mask for those same cells.

Doing this offline keeps the prediction request itself cheap: at runtime the
per-barangay summary is just a bincount over a precomputed integer array.

Only numpy, pandas and matplotlib are required. The WGS84 -> UTM 51N forward
projection is implemented inline so the script has no geospatial dependency.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

BACKEND_DIR = Path(__file__).resolve().parent.parent
ASSETS = BACKEND_DIR / "assets"
GEOJSON_PATH = ASSETS / "data" / "Sipocot_Barangays.geojson"
GRID_ALL_PATH = ASSETS / "tables" / "grid_all_cells.csv"
GRID_ACTIVE_PATH = ASSETS / "tables" / "grid_cells.csv"
OUT_META = ASSETS / "tables" / "barangay_meta.json"
OUT_INDEX = ASSETS / "tables" / "barangay_index.npz"

# A cell is treated as river channel when its contributing drainage area is at
# least CHANNEL_MIN_FLOW_ACC_LOG10 in log10(cells). With 30 m cells, 4.0 is
# 10^4 cells = 9.0 km2, a conventional channel-initiation threshold. Channel
# cells keep all their predicted values; they are only flagged so that the
# interface can distinguish in-channel water depth from inundation on land.
CHANNEL_MIN_FLOW_ACC_LOG10 = 4.0
UTM_ZONE = 51


def wgs84_to_utm(lon_deg, lat_deg, zone=UTM_ZONE):
    """Forward transverse Mercator (WGS84 -> UTM northern hemisphere), vectorised."""
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)

    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)

    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = ep2 * np.cos(lat) ** 2
    A = np.cos(lat) * (lon - lon0)

    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * lat)
        - (35 * e2**3 / 3072) * np.sin(6 * lat)
    )

    easting = k0 * N * (
        A + (1 - T + C) * A**3 / 6 + (5 - 18 * T + T**2 + 72 * C - 58 * ep2) * A**5 / 120
    ) + 500000.0
    northing = k0 * (
        M + N * np.tan(lat) * (
            A**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * ep2) * A**6 / 720
        )
    )
    return easting, northing


def feature_paths(feature):
    """Return matplotlib Paths (in UTM metres) for a Polygon/MultiPolygon feature."""
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polys = geom["coordinates"]
    else:
        return []

    paths = []
    for poly in polys:
        ring = np.asarray(poly[0], dtype=np.float64)  # exterior ring only
        ex, ny = wgs84_to_utm(ring[:, 0], ring[:, 1])
        paths.append(MplPath(np.column_stack([ex, ny])))
    return paths


def assign(points_xy, features):
    """Assign each point to a barangay index, or -1 when it falls outside all of them."""
    n = len(points_xy)
    out = np.full(n, -1, dtype=np.int16)
    remaining = np.ones(n, dtype=bool)

    for fi, feature in enumerate(features):
        if not remaining.any():
            break
        for path in feature_paths(feature):
            idx = np.flatnonzero(remaining)
            if idx.size == 0:
                break
            x0, y0, x1, y1 = path.get_extents().extents
            cand_local = (
                (points_xy[idx, 0] >= x0) & (points_xy[idx, 0] <= x1)
                & (points_xy[idx, 1] >= y0) & (points_xy[idx, 1] <= y1)
            )
            cand = idx[cand_local]
            if cand.size == 0:
                continue
            hit = path.contains_points(points_xy[cand])
            chosen = cand[hit]
            out[chosen] = fi
            remaining[chosen] = False
    return out


def main():
    print("Reading barangay boundaries...")
    geo = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    features = geo["features"]
    names = []
    for f in features:
        p = f.get("properties") or {}
        names.append(p.get("BRGY_NAME") or p.get("NAME_3") or "Unknown")
    print(f"  {len(features)} barangays")

    print("Reading full grid...")
    all_df = pd.read_csv(GRID_ALL_PATH, usecols=["row", "col", "x_coordinate", "y_coordinate"])
    all_xy = all_df[["x_coordinate", "y_coordinate"]].to_numpy(dtype=np.float64)
    print(f"  {len(all_df):,} cells in the model grid extent")

    print("Assigning every grid cell to a barangay...")
    all_idx = assign(all_xy, features)
    inside = all_idx >= 0
    totals = np.bincount(all_idx[inside], minlength=len(features)).tolist()
    sipocot_total = int(inside.sum())
    print(f"  {sipocot_total:,} cells fall inside Sipocot "
          f"({sipocot_total / len(all_df) * 100:.1f}% of the grid extent)")

    print("Assigning active model cells...")
    act_df = pd.read_csv(GRID_ACTIVE_PATH)
    act_xy = act_df[["x_coordinate", "y_coordinate"]].to_numpy(dtype=np.float64)
    act_idx = assign(act_xy, features)
    n_active_inside = int((act_idx >= 0).sum())
    print(f"  {len(act_df):,} active cells, {n_active_inside:,} inside Sipocot")

    channel = (act_df["log10_flow_accumulation"].to_numpy(dtype=np.float64)
               >= CHANNEL_MIN_FLOW_ACC_LOG10)
    print(f"  {int(channel.sum()):,} active cells flagged as river channel "
          f"(log10 flow accumulation >= {CHANNEL_MIN_FLOW_ACC_LOG10})")

    np.savez_compressed(OUT_INDEX, cell_barangay=act_idx.astype(np.int16),
                        cell_is_channel=channel)
    meta = {
        "barangays": names,
        "barangay_total_cells": totals,
        "sipocot_total_cells": sipocot_total,
        "grid_extent_total_cells": int(len(all_df)),
        "n_active_cells": int(len(act_df)),
        "n_active_cells_inside_sipocot": n_active_inside,
        "channel_min_flow_acc_log10": CHANNEL_MIN_FLOW_ACC_LOG10,
        "n_channel_cells": int(channel.sum()),
        "cell_size_m": 30,
        "note": (
            "Percentages of barangay and municipal area use barangay_total_cells and "
            "sipocot_total_cells as denominators, i.e. every grid cell inside the "
            "barangay boundary, whether or not it is an active model cell. Channel "
            "cells are flagged, never removed: their predicted values represent water "
            "depth in the river channel rather than inundation depth on land."
        ),
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_META.name} and {OUT_INDEX.name}")


if __name__ == "__main__":
    main()
