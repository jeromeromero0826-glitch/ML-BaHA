from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes_prediction import router as prediction_router
from app.api.routes_barangays import router as barangay_router
from app.core.asset_loader import load_all_assets


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model assets...")
    app.state.assets = load_all_assets()
    print("Assets loaded successfully.")
    yield


app = FastAPI(
    title="Flood Hazard Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",          # Vercel preview deployments
        "https://ml-baha.vercel.app",    # your production Vercel URL (update after deploy)
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # covers all vercel preview URLs
    allow_credentials=True,
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