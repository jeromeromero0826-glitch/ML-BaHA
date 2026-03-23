"""
routes_barangays.py
Place at: backend/app/api/routes_barangays.py

Reads Sipocot_Barangays.geojson from backend/assets/data/ and serves it
to the frontend. No geopandas required — just a plain GeoJSON file.

Make sure your GeoJSON uses EPSG:4326 (WGS84) coordinates so Leaflet
can render it correctly. Most GIS exports default to this.

Register in main.py:
    from app.api.routes_barangays import router as barangay_router
    app.include_router(barangay_router, prefix="/api")
"""

import json
import csv
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import io

router = APIRouter()

GEOJSON_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "data" / "Sipocot_Barangays.geojson"
)

GRID_ALL_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "tables" / "grid_all_cells.csv"
)

# ── Cache in memory so file is only read once per server start ────────────────
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    if not GEOJSON_PATH.exists():
        raise FileNotFoundError(
            f"GeoJSON not found at {GEOJSON_PATH}\n"
            "Save your barangay shapefile as Sipocot_Barangays.geojson "
            "in backend/assets/data/"
        )

    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Strip non-standard crs field — Leaflet doesn't need it and
    # some exporters write an OGC CRS84 object that confuses parsers
    data.pop("crs", None)

    # Auto-detect barangay name column and normalise to BRGY_NAME
    features = data.get("features", [])
    if features:
        props = features[0].get("properties") or {}
        candidates = ["BRGY_NAME", "NAME_3", "NAME", "Barangay", "BARANGAY",
                      "brgy_name", "name", "ADM4_EN", "ADM4ALT1EN"]
        name_col = next((c for c in candidates if c in props), None)
        if name_col is None and props:
            name_col = next(iter(props))  # fallback: first column

        if name_col and name_col != "BRGY_NAME":
            print(f"[barangays] Renaming '{name_col}' → 'BRGY_NAME'")
            for feat in features:
                p = feat.get("properties") or {}
                p["BRGY_NAME"] = p.get(name_col, "Unknown")

    _cache = data
    n = len(features)
    print(f"[barangays] Loaded {n} barangay features from GeoJSON.")
    return _cache


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/barangays", summary="Sipocot barangay boundaries (GeoJSON)")
def get_barangays():
    """Returns the full GeoJSON FeatureCollection for the Leaflet map."""
    try:
        return JSONResponse(content=_load())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load barangay data: {e}")


@router.get("/barangays/names", summary="Sorted list of barangay names only")
def get_barangay_names():
    """Lightweight endpoint — returns just the names for the sidebar list."""
    try:
        geojson = _load()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    names = sorted({
        f["properties"].get("BRGY_NAME", "")
        for f in geojson.get("features", [])
        if f.get("properties")
    } - {""})

    return {"count": len(names), "barangays": names}


@router.get("/grid-all-cells", summary="All spatial grid cells (active + filtered)")
def get_grid_all_cells():
    if not GRID_ALL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "grid_all_cells.csv not found. "
                "Re-run export_grid_assets.py to generate it."
            )
        )
    def iterfile():
        with open(GRID_ALL_PATH, "rb") as f:
            yield from f
    return StreamingResponse(
        iterfile(),
        media_type="text/csv",
        headers={"Content-Disposition": "inline; filename=grid_all_cells.csv"}
    )


@router.get("/grid-total", summary="Total number of grid cells in study area")
def get_grid_total():
    """
    Returns the total cell count (active + filtered + nodata) for use as
    the denominator in the map legend percentage computation.
    """
    if not GRID_ALL_PATH.exists():
        raise HTTPException(status_code=404, detail="grid_all_cells.csv not found.")
    # Count lines minus header
    count = 0
    with open(GRID_ALL_PATH, "r") as f:
        for i, _ in enumerate(f):
            if i > 0:
                count += 1
    return {"total_cells": count}