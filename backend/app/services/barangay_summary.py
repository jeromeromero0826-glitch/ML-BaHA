"""
barangay_summary.py — per-barangay hazard aggregation.

This used to run in the browser, which meant shipping the full 479k-row grid
(16.6 MB) plus the prediction CSV to every client on every prediction. On a
free-tier backend that transfer regularly failed and the Summary tab showed
nothing. The assignment of cells to barangays is static, so it is precomputed
once by scripts/build_barangay_index.py and the per-request work is reduced to
a couple of bincounts over 68k integers, which takes a few milliseconds.

Two conventions matter here:

  Denominator. Percentages are shares of barangay AREA, so the denominator is
  every 30 m grid cell inside the barangay boundary, not just the active model
  cells. A barangay that is 40% active cells and 60% filtered-out dry ground is
  reported as at most 40% hazard-classified, which is what a planner expects.

  Channel cells. Cells on the river network (high flow accumulation) legitimately
  carry several metres of water in the channel itself. They are counted like any
  other cell, but they are also reported separately so the interface can show a
  land-only maximum depth alongside the overall one instead of presenting an
  in-channel value as inundation depth on land.
"""

from typing import Dict, List

import numpy as np

# Class codes in display order. 0 is No Hazard; 1-5 are Low..Extreme.
_MIN_CODE, _MAX_CODE = 0, 5


def build_barangay_summary(
    predicted_depth: np.ndarray,
    predicted_hazard: np.ndarray,
    barangay_assets: dict,
) -> List[Dict]:
    """
    Aggregate cell-level predictions into one record per barangay.

    predicted_depth / predicted_hazard are row-aligned with grid_cells.csv,
    which is the same order the precomputed index was built in.
    """
    names: List[str] = barangay_assets["names"]
    area_cells: List[int] = barangay_assets["area_cells"]
    cell_brgy: np.ndarray = barangay_assets["cell_barangay"]
    cell_channel: np.ndarray = barangay_assets["cell_is_channel"]

    n_brgy = len(names)
    records = []

    # One pass to bucket active cells by barangay, then per-barangay bincounts.
    order = np.argsort(cell_brgy, kind="stable")
    sorted_brgy = cell_brgy[order]
    # Active cells outside every barangay carry -1 and sort to the front.
    start = int(np.searchsorted(sorted_brgy, 0, side="left"))
    bounds = np.searchsorted(sorted_brgy[start:], np.arange(n_brgy + 1), side="left") + start

    for bi in range(n_brgy):
        member = order[bounds[bi]:bounds[bi + 1]]
        total_area = int(area_cells[bi]) or 1

        codes = predicted_hazard[member]
        depths = predicted_depth[member]
        channel = cell_channel[member]

        counts = np.bincount(
            np.clip(codes.astype(np.intp), _MIN_CODE, _MAX_CODE),
            minlength=_MAX_CODE + 1,
        )
        # Cells inside the barangay that the model never evaluated are dry ground.
        inactive = total_area - int(member.size)
        counts[0] += max(inactive, 0)

        land = depths[~channel] if member.size else depths
        classified = int(counts[1:].sum())

        pcts = {str(c): round(float(counts[c]) / total_area * 100.0, 1)
                for c in range(_MIN_CODE, _MAX_CODE + 1)}

        # Dominant class is the most common hazard class, ignoring No Hazard.
        dominant = int(np.argmax(counts[1:]) + 1) if classified > 0 else 0

        records.append({
            "name":              names[bi],
            "area_cells":        total_area,
            "area_km2":          round(total_area * 0.0009, 2),
            "active_cells":      int(member.size),
            "counts":            {str(c): int(counts[c]) for c in range(_MIN_CODE, _MAX_CODE + 1)},
            "pcts":              pcts,
            "dominant":          dominant,
            "classified_pct":    round(classified / total_area * 100.0, 1),
            "max_depth_m":       round(float(depths.max()), 3) if member.size else 0.0,
            "max_depth_land_m":  round(float(land.max()), 3) if land.size else 0.0,
            "mean_depth_m":      round(float(depths.sum()) / total_area, 3) if member.size else 0.0,
            "channel_cells":     int(channel.sum()),
        })

    # Most affected first, so the table opens on what matters.
    records.sort(key=lambda r: (-r["dominant"], -r["classified_pct"], r["name"]))
    return records


def build_area_totals(
    predicted_hazard: np.ndarray,
    barangay_assets: dict,
) -> Dict:
    """
    Municipality-wide class counts expressed over Sipocot's land area.

    The legend used to divide by all 479,272 cells of the raster bounding box,
    most of which lie outside the municipality, so "No Hazard" was really
    measuring how much of the rectangle was not modelled. Here the denominator
    is the 193,718 cells inside the barangay boundaries.
    """
    cell_brgy: np.ndarray = barangay_assets["cell_barangay"]
    sipocot_total = int(barangay_assets["sipocot_total_cells"]) or 1

    inside = cell_brgy >= 0
    codes = np.clip(predicted_hazard[inside].astype(np.intp), _MIN_CODE, _MAX_CODE)
    counts = np.bincount(codes, minlength=_MAX_CODE + 1)
    counts[0] += sipocot_total - int(inside.sum())

    return {
        "denominator":       sipocot_total,
        "denominator_label": "cells within Sipocot municipal boundary",
        "area_km2":          round(sipocot_total * 0.0009, 1),
        "class_counts":      {str(c): int(counts[c]) for c in range(_MIN_CODE, _MAX_CODE + 1)},
        "class_pcts":        {str(c): round(float(counts[c]) / sipocot_total * 100.0, 1)
                              for c in range(_MIN_CODE, _MAX_CODE + 1)},
    }
