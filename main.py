"""
main.py
=======
FastAPI server for SAR-based wind field estimation.
Endpoints:
    GET /api/v1/wind-field  → wind speed + direction at a point
    GET /api/v1/wind-map    → wind vectors over a named region grid
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import logging
import os

from gee_engine import fetch_sar_image
from windmodel import get_wind_vectors, estimate_wind_direction_from_array, cmod5n_forward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SAR Wind Field API",
    description="Ocean wind field estimation using Sentinel-1 SAR imagery",
    version="1.0.0"
)

# ── ResNet model loader (optional) ──────────────────────────────────────────

MODEL_PATH = "best_resnet50_direction_model.pth"
resnet_model = None

def load_resnet():
    global resnet_model
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"No trained model found at {MODEL_PATH}. Using gradient fallback.")
        return
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        resnet_model = model
        logger.info("ResNet model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ResNet model: {e}")

load_resnet()  # called once at startup


def predict_direction_resnet(sar_array: np.ndarray) -> float:
    """Run ResNet inference on a SAR patch → wind direction (degrees)."""
    arr = sar_array.astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # 1,1,H,W
    t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
    t = t.squeeze(0).repeat(3, 1, 1).unsqueeze(0)         # 1,3,224,224
    with torch.no_grad():
        pred = resnet_model(t).squeeze().item()
    return round(float(pred) % 360, 1)


def compute_wind(lat: float, lon: float, date: str) -> dict:
    """Core logic: fetch SAR → compute wind vectors."""
    sar_data = fetch_sar_image(lat, lon, date)
    sar_array = sar_data["sar_array"]

    # Use ResNet if available, else gradient method
    if resnet_model is not None and sar_array.size > 0:
        wind_dir = predict_direction_resnet(sar_array)
        method = "ResNet-50"
    else:
        wind_dir = estimate_wind_direction_from_array(
            sar_array, sar_data["radar_look_direction"]
        )
        method = "gradient-fallback"

    wind_speed = cmod5n_forward(sar_data["mean_sigma0_db"], sar_data["incidence_angle"])

    wind_dir_rad = np.radians(wind_dir)
    u = round(-wind_speed * np.sin(wind_dir_rad), 3)
    v = round(-wind_speed * np.cos(wind_dir_rad), 3)

    return {
        "lat": lat,
        "lon": lon,
        "date": date,
        "exact_timestamp": sar_data["exact_timestamp"],
        "wind_speed_ms": wind_speed,
        "wind_dir_deg": wind_dir,
        "u_ms": u,
        "v_ms": v,
        "incidence_angle_deg": sar_data["incidence_angle"],
        "sigma0_db": sar_data["mean_sigma0_db"],
        "direction_method": method,
        "data_fusion_applied": False
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/v1/wind-field")
def wind_field(
    lat:  float = Query(..., description="Latitude  (e.g. 21.6)"),
    lon:  float = Query(..., description="Longitude (e.g. 69.6)"),
    date: str   = Query(..., description="Date YYYY-MM-DD")
):
    try:
        result = compute_wind(lat, lon, date)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        logger.exception("wind-field error")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/v1/wind-map")
def wind_map(
    region: str = Query(..., description="'gujarat' or 'tamilnadu'"),
    date:   str = Query(..., description="Date YYYY-MM-DD"),
    grid:   int = Query(4, description="Grid points per side (default 4 → 16 points)")
):
    REGIONS = {
        "gujarat":   {"lat": (20.0, 24.0), "lon": (68.0, 74.0)},
        "tamilnadu": {"lat": (8.0,  13.5), "lon": (79.0, 80.5)},
    }
    region_key = region.lower()
    if region_key not in REGIONS:
        return JSONResponse(status_code=400, content={"error": f"Unknown region '{region}'. Use 'gujarat' or 'tamilnadu'."})

    bounds = REGIONS[region_key]
    lats = np.linspace(bounds["lat"][0], bounds["lat"][1], grid).tolist()
    lons = np.linspace(bounds["lon"][0], bounds["lon"][1], grid).tolist()

    vectors = []
    errors  = []

    for lat in lats:
        for lon in lons:
            try:
                result = compute_wind(round(lat, 3), round(lon, 3), date)
                vectors.append(result)
            except Exception as e:
                errors.append({"lat": lat, "lon": lon, "error": str(e)})

    return JSONResponse(content={
        "region": region_key,
        "date": date,
        "grid_size": grid,
        "total_points": len(vectors),
        "wind_vectors": vectors,
        "errors": errors
    })


@app.get("/")
def root():
    return {"message": "SAR Wind Field API is running. Visit /docs for Swagger UI."}