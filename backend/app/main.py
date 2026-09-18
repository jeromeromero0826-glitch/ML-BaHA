from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes_prediction import router as prediction_router
from app.api.routes_barangays import router as barangay_router
from app.core.asset_loader import load_all_assets
from app.services.prediction_service import purge_output_dirs


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Leftovers from a previous process are unreachable: the scenario registry
    # lives in memory and starts empty, so nothing can link to them.
    removed = purge_output_dirs()
    if removed:
        print(f"Cleared {removed} output file(s) from a previous run.")
    print("Loading model assets...")
    app.state.assets = load_all_assets()
    print("Assets loaded successfully.")
    yield


app = FastAPI(
    title="Flood Hazard Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

# The prediction CSV and GeoJSON are highly compressible text; gzip cuts the
# barangay GeoJSON and the per-cell CSV to a fraction of their size, which
# matters a lot on a conference network talking to a free-tier instance.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://ml-ba-ha.vercel.app",   # production frontend
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # covers Vercel preview deployments
    # The API is public and stateless: no cookies, no auth headers, nothing to
    # protect per-origin. Leaving credentials on would have meant any page served
    # from any *.vercel.app domain could make credentialed calls here.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Prediction-Time-Ms"],
)

app.include_router(prediction_router, prefix="/api")
app.include_router(barangay_router, prefix="/api")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/")
def root():
    return {
        "message": "Flood Hazard Prediction API is running.",
        "docs_url": "/docs",
        "health_url": "/api/health",
        "predict_url": "/api/predict",
        "scenarios_url":  "/api/scenarios",
        "barangays_url":  "/api/barangays",
        "outputs_url":    "/outputs",
    }