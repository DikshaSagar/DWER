"""
climate_utils.py

Reusable utility functions for climate data processing.
These functions are intended to be shared across notebooks
in this repository.
"""
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

# -----------------------
# Temperature helpers
# -----------------------
def kelvin_to_celsius(da):
   
    da_c = da - 273.15
    da_c.attrs = da.attrs.copy()
    da_c.attrs["units"] = "°C"
    return da_c


# -----------------------
# Lat/lon detection & WA mask
# -----------------------

def detect_lat_lon(ds: xr.Dataset):
    """
    Detect 2-D latitude and longitude arrays in a NARCliM / WRF dataset.

    Tries ('lat','lon') then ('XLAT','XLONG').
    """
    if "lat" in ds.variables and "lon" in ds.variables:
        return ds["lat"].values, ds["lon"].values
    if "XLAT" in ds.variables and "XLONG" in ds.variables:
        return ds["XLAT"].values, ds["XLONG"].values
    raise KeyError(
        "Could not find lat/lon arrays in dataset "
        "(expected 'lat'/'lon' or 'XLAT'/'XLONG')."
    )


def mask_wa_bbox(
    ds: xr.Dataset,
    lat_name: str = "lat",
    lon_name: str = "lon",
    lon_min: float = 112,
    lon_max: float = 129,
    lat_min: float = -36,
    lat_max: float = -13,
) -> xr.DataArray:
    """
    Boolean mask of grid cells inside a WA lat/lon bounding box.

    Works with 2-D lat/lon; longitudes are normalised to 0–360.
    """
    lat2d = ds[lat_name]
    lon2d = ds[lon_name]
    lon2d360 = (lon2d + 360) % 360

    mask = (
        (lon2d360 >= lon_min) & (lon2d360 <= lon_max) &
        (lat2d >= lat_min) & (lat2d <= lat_max)
    )
    return mask


# -----------------------
# Native-grid point table
# -----------------------

def build_grid_points(ds: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame of native WRF grid points with rlat/rlon indices.

    Used when doing any region-based spatial joins (e.g. NRM regions).
    """
    lat, lon = detect_lat_lon(ds)
    rlat_idx, rlon_idx = np.indices(lat.shape)

    points = gpd.GeoDataFrame(
        {
            "rlat": rlat_idx.ravel(),
            "rlon": rlon_idx.ravel(),
        },
        geometry=gpd.points_from_xy(lon.ravel(), lat.ravel()),
        crs="EPSG:4326",
    )
    return points
