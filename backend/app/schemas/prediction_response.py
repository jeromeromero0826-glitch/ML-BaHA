from pydantic import BaseModel
from typing import List, Optional


class _FlexModel(BaseModel):
    model_config = {"extra": "ignore"}


class HazardClassCount(_FlexModel):
    code: int
    name: str
    label: str
    cell_count: int
    color: Optional[str] = None


class RainfallFeatures(_FlexModel):
    duration: float
    depth: float
    antecedent: float
    intensity: float
    total_rain: float
    antecedent_ratio: float


class PredictionSummary(_FlexModel):
    timestamp: str
    input_rainfall: RainfallFeatures
    n_cells: int
    n_flooded_cells: int
    n_no_hazard_cells: int
    max_depth_m: float
    mean_depth_all_cells_m: float
    mean_depth_flooded_cells_m: float
    hazard_class_counts: List[HazardClassCount]


class OutputFiles(_FlexModel):
    scenario_id: str
    depth_raster: str
    hazard_raster: str
    prediction_csv: str
    summary_json: str
    hazard_png: str


class MapOutputs(_FlexModel):
    hazard_png: str
    bounds: List[List[float]]
    width: int
    height: int


class PredictionResponse(_FlexModel):
    message: str
    input_rainfall: RainfallFeatures
    summary: PredictionSummary
    hazard_class_counts: List[HazardClassCount]
    outputs: OutputFiles
    map_outputs: MapOutputs