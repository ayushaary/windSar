"""
visualize_wind.py
=================
Generates wind vector + heatmap images for Gujarat and Tamil Nadu
matching research paper style (white background + coastline).
Requires: pip install matplotlib cartopy requests numpy
"""

import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("cartopy not found - using basic matplotlib (no coastline)")

API_BASE = "http://localhost:8000"

REGIONS = {
    "gujarat": {
        "date": "2024-07-16",
        "grid": 6,
        "title": "Ocean Wind Field - GUJARAT | 2024-07-16",
        "extent": [67.5, 74.5, 19.5, 24.5],
    },
    "tamilnadu": {
        "date": "2024-07-15",
        "grid": 6,
        "title": "Ocean Wind Field - TAMIL NADU | 2024-07-15",
        "extent": [78.5, 81.0, 7.5, 13.5],
    }
}


def fetch_vectors(region_name, config):
    print(f"Fetching {region_name} data from API...")
    resp = requests.get(
        f"{API_BASE}/api/v1/wind-map",
        params={"region": region_name, "date": config["date"], "grid": config["grid"]},
        timeout=600
    )
    data = resp.json()
    vectors = data["wind_vectors"]
    print(f"  Got {len(vectors)} vectors")
    return vectors


def interpolate_grid(lons, lats, values, extent, res=150):
    lon_grid = np.linspace(extent[0], extent[1], res)
    lat_grid = np.linspace(extent[2], extent[3], res)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    VALS = np.zeros_like(LON)
    for i in range(res):
        for j in range(res):
            dist = np.sqrt((lons - LON[i,j])**2 + (lats - LAT[i,j])**2)
            w = 1.0 / (dist**2 + 1e-9)
            VALS[i,j] = np.sum(w * values) / np.sum(w)
    return LON, LAT, VALS


def plot_with_cartopy(region_name, config, vectors):
    lons  = np.array([v["lon"] for v in vectors])
    lats  = np.array([v["lat"] for v in vectors])
    u     = np.array([v["u_ms"] for v in vectors])
    v_arr = np.array([v["v_ms"] for v in vectors])
    spd   = np.array([v["wind_speed_ms"] for v in vectors])

    extent = config["extent"]

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='#d4c9a8', zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=1)

    LON, LAT, SPD_GRID = interpolate_grid(lons, lats, spd, extent)
    cf = ax.contourf(LON, LAT, SPD_GRID, levels=20,
                     cmap='jet', alpha=0.75,
                     transform=ccrs.PlateCarree(), zorder=2)

    ax.quiver(lons, lats, u, v_arr,
              transform=ccrs.PlateCarree(),
              scale=60, width=0.003,
              headwidth=4, color='black', zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                        pad=0.08, shrink=0.9, aspect=30)
    cbar.set_label('Wind Speed (m/s)', fontsize=11)

    ax.set_title(config["title"], fontsize=13, fontweight='bold', pad=12)

    fname = f"wind_map_{region_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {fname}")


def plot_basic(region_name, config, vectors):
    lons  = np.array([v["lon"] for v in vectors])
    lats  = np.array([v["lat"] for v in vectors])
    u     = np.array([v["u_ms"] for v in vectors])
    v_arr = np.array([v["v_ms"] for v in vectors])
    spd   = np.array([v["wind_speed_ms"] for v in vectors])

    extent = config["extent"]
    fig, ax = plt.subplots(figsize=(8, 10))

    LON, LAT, SPD_GRID = interpolate_grid(lons, lats, spd, extent)
    cf = ax.contourf(LON, LAT, SPD_GRID, levels=20, cmap='jet', alpha=0.8)
    ax.quiver(lons, lats, u, v_arr, color='black',
              scale=60, width=0.003, headwidth=4)

    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1, shrink=0.9)
    cbar.set_label('Wind Speed (m/s)', fontsize=11)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(config["title"], fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    fname = f"wind_map_{region_name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved {fname}")


if __name__ == "__main__":
    for region_name, config in REGIONS.items():
        try:
            vectors = fetch_vectors(region_name, config)
            if HAS_CARTOPY:
                plot_with_cartopy(region_name, config, vectors)
            else:
                plot_basic(region_name, config, vectors)
        except Exception as e:
            print(f"ERROR for {region_name}: {e}")

    print("\nDone! Check wind_map_gujarat.png and wind_map_tamilnadu.png")