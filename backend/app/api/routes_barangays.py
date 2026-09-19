"""
routes_barangays.py — Sipocot barangay boundaries for the Leaflet map.

Reads Sipocot_Barangays.geojson from backend/assets/data/ once per process and
serves it. The file is EPSG:4326 so Leaflet can draw it directly.

The two grid endpoints that used to live here are gone. /grid-all-cells streamed
a 16 MB CSV and /grid-total re-counted 479,272 lines on every call; both existed
only to let the browser rebuild the barangay summary itself, which the server has
computed since the summary moved into the prediction response.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

GEOJSON_PATH = (
    Path(__file__).resolve().parents[2] / "assets" / "data" / "Sipocot_Barangays.geojson"
)

# Boundaries change on the order of years, not minutes. Without this the CDN in
# front of the instance marks every response DYNAMIC and each visitor both
# re-downloads 39 KB and wakes a sleeping instance to get it.
BOUNDARY_CACHE_CONTROL = "public, max-age=86400, stale-while-revalidate=604800"

# ── Cache in memory so the file is only read once per server start ────────────
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
    print(f"[barangays] Loaded {len(features)} barangay features from GeoJSON.")
    return _cache


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/barangays", summary="Sipocot barangay boundaries (GeoJSON)")
def get_barangays():
    """Returns the full GeoJSON FeatureCollection for the Leaflet map."""
    try:
        return JSONResponse(
            content=_load(),
            headers={"Cache-Control": BOUNDARY_CACHE_CONTROL},
        )
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

    return JSONResponse(
        content={"count": len(names), "barangays": names},
        headers={"Cache-Control": BOUNDARY_CACHE_CONTROL},
    )
