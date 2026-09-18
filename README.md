# ML-BaHa

**Machine Learning-Based Hazard prediction tool** — a browser-based flood hazard
prediction and mapping application for Sipocot, Camarines Sur, Philippines.

Live at **[ml-ba-ha.vercel.app](https://ml-ba-ha.vercel.app)**.

Enter three rainfall values (storm depth, duration, antecedent rainfall) and the
application returns a five-class flood hazard map and a per-barangay summary for
all 37 barangays of Sipocot in a few seconds, against roughly six to eight hours
for the equivalent HEC-RAS 2D simulation.

## How it works

A tuned **XGBoost** surrogate model, trained on 109 HEC-RAS 2D Rain-on-Grid
simulations, predicts flood depth for each of 68,276 active 30 m grid cells from
12 features: six derived from the rainfall inputs and six static terrain and
spatial attributes (coordinates, elevation, slope, log₁₀ flow accumulation, TWI).
Predicted depths are classified into five hazard tiers following the DENR-MGB
depth scheme.

## Layout

```
backend/            FastAPI service (deployed on Render)
  app/api/          route handlers
  app/core/         configuration and startup asset loading
  app/services/     prediction pipeline, hazard classification, raster output,
                    per-barangay aggregation
  assets/           model, spatial grid, barangay boundaries, precomputed index
  scripts/          offline asset generation
frontend/           React + Vite single-page app (deployed on Vercel)
DEPLOY.md           deployment order, commands and a live smoke-test checklist
```

## Running locally

**Backend**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Serves on `http://127.0.0.1:8000`; interactive API docs at `/docs`.

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Serves on `http://127.0.0.1:5173` and reads `.env.development`, which points at
the local backend.

## Regenerating the barangay index

The assignment of grid cells to barangays is static, so it is precomputed rather
than recalculated per request. Re-run it only when the boundaries or the spatial
grid change:

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python scripts/build_barangay_index.py
```

This writes `assets/tables/barangay_meta.json` and `barangay_index.npz`, both of
which are committed. The server refuses to start without them.

## Conventions worth knowing

**Percentages are shares of land area.** The denominator is every 30 m cell inside
the relevant boundary, 193,718 cells (174.3 km²) for the municipality, not just
the cells the model evaluates. Cells filtered out as permanently dry during
training are included and counted as No Hazard, so classes always sum to 100% of
area. A consequence: No Hazard sits near 65% for any rainfall, because roughly two
thirds of Sipocot's cells were filtered out.

**River channel cells are flagged, not removed.** Cells whose contributing drainage
area reaches 10⁴ cells (about 9 km²) carry water depth in the channel itself rather
than inundation depth on land, and during a large storm can legitimately exceed the
Extreme threshold by a wide margin. They are counted in every class total, and the
interface reports a separate land-only maximum alongside the overall one.

## Citation

Romero, J.P., Duka, M.A., Lampayan, R.M., Saludes, R.B. *ML-BaHa: Operationalizing
Surrogate Machine Learning for Real-Time Flood Hazard Mapping in Camarines Sur,
Philippines.* Proceedings of the International Exchange and Innovation Conference
on Engineering & Sciences (IEICES), 2026.

## Acknowledgments

Field survey data from the Sipocot MDRRMO. Rainfall data from PAGASA and the Bicol
River Basin Flood Forecasting and Warning Center. Terrain and spatial data from
NAMRIA and BSWM.
