"""
gee_engine.py
=============
Fetches Sentinel-1 SAR backscatter and incidence angle
from Google Earth Engine for a given lat/lon and date.
"""

import ee
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Initialize GEE (assumes authentication is done)
try:
    ee.Initialize(project='quiet-coda-470111-b3')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='quiet-coda-470111-b3')

def fetch_sar_image(lat: float, lon: float, target_date: str) -> dict:
    """
    Fetches Sentinel-1 SAR data for a given location and date.

    Returns a dict with:
        - sar_array: 2D numpy array of VV backscatter (linear scale)
        - mean_sigma0_db: mean backscatter in dB
        - incidence_angle: mean incidence angle in degrees
        - exact_timestamp: ISO-8601 timestamp of the actual image used
        - radar_look_direction: approx look direction (90 degrees by default for S1)
    """
    point = ee.Geometry.Point([lon, lat])

    # Search ±15 days around the target date to find the nearest pass
    start = ee.Date(target_date).advance(-15, 'day')
    end   = ee.Date(target_date).advance(15, 'day')

    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .sort('system:time_start')  # nearest pass
    )

    size = collection.size().getInfo()
    if size == 0:
        raise ValueError(f"No Sentinel-1 data found near {lat},{lon} around {target_date}")

    image = collection.first()

    # Get the timestamp
    timestamp_ms = image.get('system:time_start').getInfo()
    from datetime import datetime, timezone
    exact_ts = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()

    # Define a 20km x 20km region around the point
    region = point.buffer(10000).bounds()

    # Sample VV band
    vv_band = image.select('VV')
    vv_info = vv_band.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=100,
        maxPixels=1e9
    ).getInfo()
    mean_sigma0_db = float(vv_info.get('VV', -15.0))

    # Sample incidence angle
    angle_info = image.select('angle').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=100,
        maxPixels=1e9
    ).getInfo()
    incidence_angle = float(angle_info.get('angle', 35.0))

    # Get a small patch as array for the ResNet (64x64 pixels)
    patch = vv_band.sampleRectangle(region=point.buffer(1600).bounds(), defaultValue=-20)
    arr = np.array(patch.get('VV').getInfo(), dtype=np.float32)

    logger.info(f"GEE fetch OK | ts={exact_ts} | sigma0={mean_sigma0_db:.2f}dB | angle={incidence_angle:.1f}°")

    return {
        "sar_array": arr,
        "mean_sigma0_db": float(mean_sigma0_db),
        "incidence_angle": float(incidence_angle),
        "exact_timestamp": exact_ts,
        "radar_look_direction": 90.0
    }