"""
validate.py
===========
Compares your API's wind speed/direction against ERA5 reference data.
Produces a CSV table for submission.
"""

import requests
import csv
import math

# Test points: (lat, lon, date, era5_speed, era5_dir)
# Get ERA5 reference from Open-Meteo archive (free)
TEST_CASES = [
    (21.5, 69.0, "2024-07-16"),
    (22.0, 69.5, "2024-07-16"),
    (10.0, 80.0, "2024-07-15"),
    (11.0, 79.8, "2024-07-15"),
]

API_BASE = "http://localhost:8000"

def get_era5_reference(lat, lon, date):
    """Fetch ERA5 wind from Open-Meteo as ground truth."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=wind_speed_10m,wind_direction_10m"
    )
    r = requests.get(url, timeout=15).json()
    speed = r['hourly']['wind_speed_10m'][12]   # noon
    direc = r['hourly']['wind_direction_10m'][12]
    return speed, direc

def angular_error(pred, true):
    diff = abs(pred - true) % 360
    return min(diff, 360 - diff)

results = []
for lat, lon, date in TEST_CASES:
    # Call your API
    resp = requests.get(f"{API_BASE}/api/v1/wind-field", 
                        params={"lat": lat, "lon": lon, "date": date})
    api = resp.json()
    
    # Get ERA5 reference
    era5_speed, era5_dir = get_era5_reference(lat, lon, date)
    
    if 'error' in api:
        print(f"  API ERROR for {lat},{lon}: {api['error']}")
        continue
    speed_err = abs(api['wind_speed_ms'] - era5_speed)
    dir_err   = angular_error(api['wind_dir_deg'], era5_dir)
    
    results.append({
        "lat": lat, "lon": lon, "date": date,
        "api_speed_ms": api['wind_speed_ms'],
        "era5_speed_ms": round(era5_speed, 2),
        "speed_error": round(speed_err, 2),
        "api_dir_deg": api['wind_dir_deg'],
        "era5_dir_deg": round(era5_dir, 1),
        "dir_error_deg": round(dir_err, 1),
        "data_fusion": api['data_fusion_applied'],
    })
    print(f"  {lat},{lon} {date} → speed_err={speed_err:.2f} dir_err={dir_err:.1f}°")

# Compute summary stats
n = len(results)
mae_speed = sum(r['speed_error'] for r in results) / n
mae_dir   = sum(r['dir_error_deg'] for r in results) / n
rmse_speed = math.sqrt(sum(r['speed_error']**2 for r in results) / n)

print(f"\nMAE speed: {mae_speed:.2f} m/s | RMSE: {rmse_speed:.2f} | MAE dir: {mae_dir:.1f}°")

# Save CSV
with open("Validation_Results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print("Saved Validation_Results.csv")