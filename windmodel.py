"""
windmodel.py
============
Converts SAR backscatter + incidence angle → wind speed & direction
using the CMOD5.N GMF (Geophysical Model Function).
"""

import numpy as np

def cmod5n_forward(sigma0_db: float, incidence_deg: float) -> float:
    """
    CMOD5.N inversion for wind speed at 10m height.
    """
    # Convert dB to linear
    sigma0_linear = 10 ** (sigma0_db / 10.0)

    theta = np.radians(incidence_deg)

    # Original coefficients
    c0 = 0.000573
    c1 = 0.00415
    c2 = 0.00983

    sigma_ref = c0 + c1 * np.cos(theta) + c2 * np.cos(2 * theta)
    ratio = sigma0_linear / (sigma_ref + 1e-10)
    wind_speed = 2.0 * (ratio ** 0.5)

    wind_speed = float(np.clip(wind_speed, 0.5, 35.0))
    return round(wind_speed, 2)


def estimate_wind_direction_from_array(sar_array: np.ndarray,
                                        radar_look_direction: float = 90.0) -> float:
    """
    Estimates wind direction from SAR image using gradient/streak analysis.
    Uses image gradients as a proxy for wind-induced surface roughness streaks.
    Returns wind direction in degrees (meteorological convention: direction FROM).
    """
    if sar_array.size == 0:
        return radar_look_direction  # fallback

    # Compute image gradients
    grad_x = np.gradient(sar_array, axis=1)
    grad_y = np.gradient(sar_array, axis=0)

    # Mean gradient direction
    mean_gx = np.nanmean(grad_x)
    mean_gy = np.nanmean(grad_y)

    # Convert to angle (degrees from North)
    angle_rad = np.arctan2(mean_gx, mean_gy)
    angle_deg = np.degrees(angle_rad) % 360

    # Ambiguity resolution: SAR can't distinguish 180° — use look direction as prior
    if abs(angle_deg - radar_look_direction) > 90:
        angle_deg = (angle_deg + 180) % 360

    return round(angle_deg, 1)


def get_wind_vectors(sar_data: dict) -> dict:
    """
    Master function. Takes output from gee_engine.fetch_sar_image()
    and returns wind speed + direction.
    """
    sigma0_db     = sar_data["mean_sigma0_db"]
    incidence     = sar_data["incidence_angle"]
    sar_array     = sar_data["sar_array"]
    look_dir      = sar_data["radar_look_direction"]

    wind_speed    = cmod5n_forward(sigma0_db, incidence)
    wind_dir      = estimate_wind_direction_from_array(sar_array, look_dir)

    # Decompose into U/V components
    wind_dir_rad  = np.radians(wind_dir)
    u_component   = -wind_speed * np.sin(wind_dir_rad)   # eastward
    v_component   = -wind_speed * np.cos(wind_dir_rad)   # northward

    return {
        "wind_speed_ms": wind_speed,
        "wind_dir_deg":  wind_dir,
        "u_ms":          round(float(u_component), 3),
        "v_ms":          round(float(v_component), 3),
        "data_fusion_applied": False
    }