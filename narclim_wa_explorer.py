"""
narclim_wa_explorer.py

Reusable module for the NARCliM 2.0 Western Australia Climate Explorer.

Usage from a notebook
---------------------

from narclim_wa_explorer import build_wa_explorer_ui
ui = build_wa_explorer_ui()
display(ui)
"""

import warnings
warnings.filterwarnings("ignore")

import re
import json
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import xarray as xr
import plotly.express as px
import matplotlib.pyplot as plt
import ipywidgets as widgets
import geopandas as gpd

from climate_utils import kelvin_to_celsius  # expects kelvin_to_celsius(da) -> °C

# =============================================================================
# GLOBALS
# =============================================================================

ds = None          # current xarray.Dataset
df_stats = None    # GeoDataFrame with polygons + mean
var_names = None   # list of variables discovered
gdf_regions = None # AACC regions GeoDataFrame (WA only)

# These will be defined in build_wa_explorer_ui()
variable_dd = None
year_slider = None
run_btn = None
output = None
region_sel = None
ts_btn = None
ui = None

# =============================================================================
# THREDDS HELPERS
# =============================================================================

NS = {
    "cat": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0",
    "xlink": "http://www.w3.org/1999/xlink",
}

def catalog_xml_from_opendap_dir(opendap_dir_url: str) -> str:
    """Convert a dodsC directory URL to the corresponding THREDDS catalog.xml."""
    if "/thredds/dodsC/" in opendap_dir_url:
        cat = opendap_dir_url.replace("/thredds/dodsC/", "/thredds/catalog/")
    else:
        cat = opendap_dir_url
    return cat.rstrip("/") + "/catalog.xml"

def list_subfolders_from_catalog(catalog_xml_url: str):
    """List sub-catalogs (variables) from a THREDDS catalog XML."""
    xml = requests.get(catalog_xml_url, timeout=60).text
    root = ET.fromstring(xml)
    out = []
    for cr in root.findall(".//cat:catalogRef", NS):
        name = cr.get("name") or cr.get("{http://www.w3.org/1999/xlink}title")
        href = cr.get("{http://www.w3.org/1999/xlink}href")
        if name and href:
            out.append((name, urljoin(catalog_xml_url, href)))
    return out

def list_files_from_catalog(catalog_xml_url: str):
    """List datasets (files) from a THREDDS catalog XML."""
    xml = requests.get(catalog_xml_url, timeout=60).text
    root = ET.fromstring(xml)
    return [
        (ds_el.get("name"), ds_el.get("urlPath"))
        for ds_el in root.findall(".//cat:dataset[@urlPath]", NS)
    ]

def opendap_url_from_urlpath(urlpath: str) -> str:
    """Build full OPeNDAP URL from THREDDS urlPath."""
    return "https://dapds00.nci.org.au/thredds/dodsC/" + urlpath.lstrip("/")

def pick_file_by_timerange(files, start_yyyymm: str, end_yyyymm: str):
    """
    Given (name, urlPath) list and a year-month range, pick the file
    whose [start,end] overlaps that range, else the last one.
    """
    rx = re.compile(r"_(\d{6})-(\d{6})\.nc$")
    parsed = []
    for name, urlpath in files:
        m = rx.search(name or "")
        if m:
            parsed.append((m.group(1), m.group(2), name, urlpath))

    if not parsed:
        raise ValueError("No parsable files found in catalog")

    overlapping = [
        p for p in parsed
        if not (p[1] < start_yyyymm or p[0] > end_yyyymm)
    ]
    if overlapping:
        overlapping.sort(key=lambda x: x[1])
        return overlapping[-1][2], overlapping[-1][3]

    parsed.sort(key=lambda x: x[1])
    return parsed[-1][2], parsed[-1][3]

def get_available_years_for_variable(variable: str):
    """
    Inspect the THREDDS 'latest' catalog for a variable and
    return a sorted list of years actually present in the file names.

    Assumes filenames contain _YYYYMM-YYYYMM.nc at the end.
    """
    latest_dir = (
        f"{BASE}/"
        f"{CONFIG['domain']}/"
        f"{CONFIG['org']}/"
        f"{CONFIG['gcm']}/"
        f"{CONFIG['exp']}/"
        f"{CONFIG['variant']}/"
        f"{CONFIG['rcm']}/"
        f"{CONFIG['version']}/"
        f"{CONFIG['frequency']}/"
        f"{variable}/latest/"
    )
    cat_url = catalog_xml_from_opendap_dir(latest_dir)
    files = list_files_from_catalog(cat_url)
    if not files:
        return []

    rx = re.compile(r"_(\d{4})(\d{2})-(\d{4})(\d{2})\.nc$")
    years = set()
    for name, _ in files:
        m = rx.search(name or "")
        if not m:
            continue
        y1 = int(m.group(1))
        y2 = int(m.group(3))
        for y in range(y1, y2 + 1):
            years.add(y)

    return sorted(years)

# =============================================================================
# DATASET CONFIG
# =============================================================================

BASE = "https://dapds00.nci.org.au/thredds/dodsC/zz63/NARCliM2-0/output-CMIP6/DD"
CONFIG = {
    "domain": "AUS-18",
    "org": "NSW-Government",
    "gcm": "ACCESS-ESM1-5",
    "exp": "ssp126",             # projection experiment
    "variant": "r6i1p1f1",
    "rcm": "NARCliM2-0-WRF412R3",
    "version": "v1-r1",
    "frequency": "mon",
}

WA_BOUNDS = {
    "lon": (110, 130),
    "lat": (-36, -12),
}

# Path to your AACC regions shapefile (GDA94, EPSG:4283)
SHAPEFILE_PATH = "aacc_shp/AACC_Data_Warehouse_Regions_simplify5.shp"

# Use shapefile Region_Nam values directly as region labels
UI_REGIONS = [
    "East Kimberley",
    "West Kimberley",
    "Goldfields-Esperance",
    "Midwest-Gascoyne",
    "Perth",
    "Southern",
    "Pilbara",
    "Wheatbelt",
]

# =============================================================================
# REGION SHAPEFILE LOADING
# =============================================================================

def load_aacc_regions():
    """Load AACC regions shapefile and prepare for joins."""
    global gdf_regions

    if gdf_regions is not None:
        return gdf_regions

    print(f"🗺️ Loading AACC regions from: {SHAPEFILE_PATH}")
    gdf = gpd.read_file(SHAPEFILE_PATH)

    # Fix CRS and geometries
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4283)
    elif gdf.crs.to_epsg() != 4283:
        gdf = gdf.to_crs(epsg=4283)
    gdf = gdf.to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].buffer(0)

    if "Region_Nam" not in gdf.columns:
        raise KeyError("Expected 'Region_Nam' column in AACC shapefile")

    # Use Region_Nam directly
    gdf["ui_region"] = gdf["Region_Nam"]

    # Keep only the regions we want in the UI
    gdf = gdf[gdf["ui_region"].isin(UI_REGIONS)].copy()

    print("✅ Regions loaded:", sorted(gdf["ui_region"].unique()))
    gdf_regions = gdf
    return gdf_regions

# =============================================================================
# CORE DATA FUNCTIONS
# =============================================================================

def open_selected_dataset(variable: str, year: str):
    """
    Open NARCliM dataset for a single year, subset to WA,
    convert temps to °C, and load.
    """
    global ds

    latest_dir = (
        f"{BASE}/"
        f"{CONFIG['domain']}/"
        f"{CONFIG['org']}/"
        f"{CONFIG['gcm']}/"
        f"{CONFIG['exp']}/"
        f"{CONFIG['variant']}/"
        f"{CONFIG['rcm']}/"
        f"{CONFIG['version']}/"
        f"{CONFIG['frequency']}/"
        f"{variable}/latest/"
    )
    cat_url = catalog_xml_from_opendap_dir(latest_dir)
    files = list_files_from_catalog(cat_url)
    if not files:
        raise RuntimeError("No files found in THREDDS catalog")

    start_yyyymm = f"{year}01"
    end_yyyymm   = f"{year}12"

    name, urlpath = pick_file_by_timerange(files, start_yyyymm, end_yyyymm)
    opendap_url = opendap_url_from_urlpath(urlpath)

    print(f"📂 Loading {variable} ({year}): {name}")
    # Prefer pydap if installed, otherwise fall back to netcdf4
    try:
        ds_new = xr.open_dataset(opendap_url, engine="pydap")
    except ValueError as exc:
        # xarray raises ValueError when the engine is not recognized
        if "unrecognized engine 'pydap'" in str(exc):
            ds_new = xr.open_dataset(opendap_url, engine="netcdf4")
        else:
            raise

    year_int = int(year)
    time_mask = ds_new.time.dt.year == year_int
    lon_mask = (ds_new.lon >= WA_BOUNDS["lon"][0]) & (ds_new.lon <= WA_BOUNDS["lon"][1])
    lat_mask = (ds_new.lat >= WA_BOUNDS["lat"][0]) & (ds_new.lat <= WA_BOUNDS["lat"][1])

    ds_new = ds_new.where(time_mask & lon_mask & lat_mask, drop=True).load()

    if ds_new.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No data available in {name} for year {year} within WA bounds."
        )

    if variable.startswith("tas"):
        ds_new[variable] = kelvin_to_celsius(ds_new[variable])

    ds = ds_new
    print(f"✅ Loaded dataset dims: {dict(ds.dims)}")
    print(f"   Units for {variable}: {ds[variable].attrs.get('units', 'unknown')}")
    return ds

def compute_stats(variable: str):
    """
    Compute mean over time for selected variable
    aggregated by AACC polygon regions; store in df_stats.
    """
    global df_stats

    if ds is None:
        raise RuntimeError("Dataset not loaded")

    gdf = load_aacc_regions()
    da = ds[variable]

    if da.sizes.get("time", 0) == 0:
        raise ValueError("Dataset has no time steps; check year selection.")

    print("🔢 Computing regional mean using AACC polygons...")

    # Build point grid of model cells (center of each cell)
    lat = ds["lat"]
    lon = ds["lon"]
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    else:
        lat2d = lat.values
        lon2d = lon.values

    mask_any = np.any(~np.isnan(da.values), axis=0)
    iy, ix = np.where(mask_any)

    points = gpd.GeoDataFrame(
        {
            "rlat": iy.astype(int),
            "rlon": ix.astype(int),
            "lat": lat2d[iy, ix],
            "lon": lon2d[iy, ix],
        },
        geometry=gpd.points_from_xy(lon2d[iy, ix], lat2d[iy, ix]),
        crs="EPSG:4326",
    )

    # Spatial join: assign each grid point to a region polygon
    points_regions = gpd.sjoin(
        points, gdf[["ui_region", "geometry"]],
        how="inner", predicate="within"
    )
    if points_regions.empty:
        raise RuntimeError("No grid points found inside AACC polygons – check bounds/CRS.")

    # Time mean per cell
    data = da.values  # (time, y, x)
    mean_2d = np.nanmean(data, axis=0)

    points_regions["mean_cell"] = mean_2d[points_regions["rlat"], points_regions["rlon"]]

    # Aggregate over cells per region (mean only)
    agg = (
        points_regions.groupby("ui_region")[["mean_cell"]]
        .mean()
        .reset_index()
        .rename(columns={"ui_region": "region", "mean_cell": "mean"})
    )

    # Merge back to polygons (GeoDataFrame)
    gdf_poly = gdf.copy().rename(columns={"ui_region": "region"})
    df_stats = gdf_poly.merge(agg, on="region", how="inner")

    n_regions = df_stats["region"].nunique()
    print(
        f"✅ Stats ready for {n_regions} regions: "
        f"{', '.join(sorted(df_stats['region'].unique()))}"
    )

def extract_ts_for_region(region_name: str, variable: str):
    """
    Extract time series as region-average over polygon for given region.
    """
    if ds is None:
        raise RuntimeError("Dataset not loaded")

    gdf = load_aacc_regions()
    region_poly = gdf.loc[gdf["ui_region"] == region_name]
    if region_poly.empty:
        raise ValueError(f"No polygon found for region '{region_name}'")

    lat = ds["lat"]
    lon = ds["lon"]
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon.values, lat.values)
    else:
        lat2d = lat.values
        lon2d = lon.values

    points = gpd.GeoDataFrame(
        {
            "rlat": np.arange(lat2d.shape[0]).repeat(lat2d.shape[1]),
            "rlon": np.tile(np.arange(lat2d.shape[1]), lat2d.shape[0]),
            "lat": lat2d.ravel(),
            "lon": lon2d.ravel(),
        },
        geometry=gpd.points_from_xy(lon2d.ravel(), lat2d.ravel()),
        crs="EPSG:4326",
    )

    inside = gpd.sjoin(points, region_poly[["geometry"]],
                       how="inner", predicate="within")
    if inside.empty:
        raise RuntimeError(f"No model grid cells inside polygon for region '{region_name}'")

    da = ds[variable]
    ts_vals = []
    for t_idx in range(da.sizes["time"]):
        slice_2d = da.isel(time=t_idx).values
        vals = slice_2d[inside["rlat"], inside["rlon"]]
        ts_vals.append(np.nanmean(vals))

    ts = xr.DataArray(
        data=np.array(ts_vals),
        coords={"time": ds["time"].values},
        dims=["time"],
        name=f"{variable}_region_{region_name}",
        attrs=da.attrs,
    )
    return ts

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def _df_stats_to_geojson():
    """Convert df_stats (GeoDataFrame) to GeoJSON feature collection for Plotly."""
    if df_stats is None:
        raise RuntimeError("Stats not computed")
    gdf = df_stats.set_geometry("geometry")
    geojson = json.loads(gdf.to_json())
    return geojson

def plot_map(variable: str):
    """
    Choropleth map: show mean over AACC polygons (WA-only).
    Hover shows region and mean value.
    """
    if df_stats is None:
        raise RuntimeError("Stats not computed")

    units = ds[variable].attrs.get("units", "units")
    geojson = _df_stats_to_geojson()

    fig = px.choropleth(
        df_stats,
        geojson=geojson,
        locations="region",
        featureidkey="properties.region",
        color="mean",
        color_continuous_scale="Viridis",
        hover_name="region",
        hover_data={"mean": ":.2e"},
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=f"{variable.upper()} mean over Western Australia ({year_slider.value})",
        coloraxis_colorbar=dict(title=f"Mean ({units})"),
        height=520,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor="#dfe4ea",
        plot_bgcolor="#dfe4ea",
    )

    fig.show()

def plot_timeseries(b=None):
    """Plot region-mean time series for currently selected regions."""
    if ds is None or df_stats is None:
        print("❌ Please run the analysis first (map) before plotting time series.")
        return

    selected_regions = region_sel.value
    if not selected_regions:
        print("⚠️ Please select one or more regions from the list.")
        return

    variable = variable_dd.value
    units = ds[variable].attrs.get("units", "units")

    fig, ax = plt.subplots(figsize=(12, 5))

    for region_name in selected_regions:
        ts = extract_ts_for_region(region_name, variable)
        ts.plot(ax=ax, label=region_name, linewidth=2)

    ax.set_title(
        f"{variable.upper()} time series - selected WA regions ({year_slider.value})",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel(units, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=10)
    plt.tight_layout()
    plt.show()

# =============================================================================
# UI BUILDER
# =============================================================================

def _scan_variables():
    """Scan THREDDS for available variables once."""
    global var_names

    print("🔍 Scanning NARCliM 2.0 variables...")
    var_root = (
        f"{BASE}/"
        f"{CONFIG['domain']}/"
        f"{CONFIG['org']}/"
        f"{CONFIG['gcm']}/"
        f"{CONFIG['exp']}/"
        f"{CONFIG['variant']}/"
        f"{CONFIG['rcm']}/"
        f"{CONFIG['version']}/"
        f"{CONFIG['frequency']}/"
    )
    var_cat = catalog_xml_from_opendap_dir(var_root)
    var_names = sorted([name for name, _ in list_subfolders_from_catalog(var_cat)])
    print(f"✅ Found {len(var_names)} variables. Example: {var_names[:5]}")

def build_wa_explorer_ui():
    """
    Build and return the WA climate explorer UI widget.

    Typical notebook usage:
        from narclim_wa_explorer import build_wa_explorer_ui
        ui = build_wa_explorer_ui()
        display(ui)
    """
    global variable_dd, year_slider, run_btn, output, region_sel, ts_btn, ui

    # Ensure regions are loaded once
    load_aacc_regions()

    if var_names is None:
        _scan_variables()

    # Widgets
    variable_dd = widgets.Dropdown(
        options=var_names,
        value="tasmax" if "tasmax" in var_names else var_names[0],
        description="Variable:",
        layout=widgets.Layout(width="220px"),
    )

    year_slider = widgets.SelectionSlider(
        options=["2100"],   # temporary, updated by update_year_slider()
        value="2100",
        description="Year:",
        layout=widgets.Layout(width="220px"),
    )

    run_btn = widgets.Button(
        description="🚀 ANALYZE",
        button_style="success",
        layout=widgets.Layout(width="140px"),
    )

    output = widgets.Output(layout=widgets.Layout(height="550px"))

    region_sel = widgets.SelectMultiple(
        options=UI_REGIONS,
        value=["Perth", "Southern"],
        description="WA regions:",
        rows=8,
        layout=widgets.Layout(width="260px"),
    )

    ts_btn = widgets.Button(
        description="📈 TIMESERIES",
        button_style="info",
        layout=widgets.Layout(width="140px"),
    )

    # Widget-dependent helpers
    def update_year_slider(*args):
        var = variable_dd.value
        years = get_available_years_for_variable(var)
        if not years:
            year_slider.options = ["2100"]
            year_slider.value = "2100"
            print(f"⚠️ No years discovered for {var} in 'latest' catalog")
            return
        year_strs = [str(y) for y in years]
        year_slider.options = year_strs
        year_slider.value = year_strs[0]

    variable_dd.observe(update_year_slider, names="value")
    update_year_slider()

    def on_run(b):
        with output:
            from IPython.display import clear_output as _clear_output
            _clear_output()
            year = year_slider.value
            print(f"⏳ Loading and analysing NARCliM 2.0 data for {year}...")
            try:
                open_selected_dataset(variable_dd.value, year)
                compute_stats(variable_dd.value)
                plot_map(variable_dd.value)
                print(
                    "\n✅ Map ready.\n"
                    "1) Hover regions for mean values\n"
                    "2) Select regions on the left and click TIMESERIES"
                )
            except Exception as exc:
                print(f"❌ Error during analysis: {exc}")

    def on_timeseries(b):
        with output:
            print("\n📈 Generating time series...")
            try:
                plot_timeseries()
            except Exception as exc:
                print(f"❌ Error plotting time series: {exc}")

    run_btn.on_click(on_run)
    ts_btn.on_click(on_timeseries)

    # UI layout
    ui = widgets.VBox(
        [
            widgets.HTML("<h3>NARCliM 2.0 WA Climate Explorer</h3>"),
            widgets.HBox([variable_dd, year_slider, run_btn]),
            widgets.HTML("<b>Step 2:</b> Select WA regions and click TIMESERIES"),
            widgets.HBox([region_sel, ts_btn]),
            output,
        ],
        layout=widgets.Layout(
            border="2px solid #007acc",
            padding="12px",
            width="100%",
            margin="8px 0",
        ),
    )

    return ui
