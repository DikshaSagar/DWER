# User Guide  
WA Climate Notebook Collection (NARCliM 2.0)

---

## 1. Purpose and scope

This notebook collection demonstrates a complete, reproducible workflow for working with NARCliM 2.0 regional climate model data over Western Australia using Python and Jupyter notebooks.

It is designed to help you:

- Access NARCliM 2.0 data remotely via NCI THREDDS / OPeNDAP.  
- Subset to Western Australia and official WA NRM regions.  
- Regrid model output to regular latitude–longitude grids for mapping and GIS.  
- Generate maps, time series, and regional climate summaries.  
- Explore results interactively in a notebook-based “dashboard”.

Although the examples use NARCliM 2.0, the structure can be reused for other regional climate datasets with minimal changes to data paths and variable names.

---

## 2. How to run the toolkit

### 2.1 Online (e.g. Binder or hosted Jupyter)

If you provide a Binder or similar badge in the repository, users can:

1. Open the GitHub repository containing these notebooks.  
2. Click the “Launch Binder” badge.  
3. Wait for the JupyterLab environment to start.  
4. Open any notebook (01–06) from the file browser.  
5. Run cells with `Shift+Enter`.

Binder-style environments are ideal for:

- Demonstrations and training.  
- Quick exploration of variables and plots.  
- Sharing reproducible examples.

They are not ideal for:

- Large downloads or long regridding jobs.  
- Long-running analyses—sessions are temporary and can time out.

Always download important outputs (plots, NetCDF, CSV) before closing the session.

### 2.2 Local setup using Miniforge and mamba

For sustained use and larger analyses, run the notebooks locally using a dedicated Conda environment via **Miniforge + mamba**.

#### 2.2.1 Why this setup?

- Lightweight and free.  
- Uses `conda-forge` by default (good for scientific Python).  
- `mamba` solves environments faster and with fewer conflicts than `conda`.  
- Works on macOS (Intel and Apple Silicon), Windows, and Linux.

#### 2.2.2 Install Miniforge (one-time)

1. Go to: <https://github.com/conda-forge/miniforge>.  
2. Download the installer for your OS:

   | Operating system        | Installer                     |
   |-------------------------|------------------------------|
   | macOS (Apple Silicon)   | `Miniforge3-MacOSX-arm64.sh` |
   | macOS (Intel)           | `Miniforge3-MacOSX-x86_64.sh`|
   | Windows                 | `Miniforge3-Windows-x86_64.exe` |
   | Linux                   | `Miniforge3-Linux-x86_64.sh` |

3. Run the installer and accept the default options.

#### 2.2.3 Verify installation

Open a terminal (or Anaconda Prompt on Windows) and run:

```bash
conda --version
mamba --version
```
If both commands print version numbers, Miniforge is installed correctly.

#### 2.2.4 Create and activate an environment
Create a dedicated environment for this project:

```bash
mamba create -n climate-env python=3.10
conda activate climate-env
```
Your prompt should show (climate-env) once activated.

#### 2.2.5 Install required packages
Install the core packages used in the notebooks:

```bash
mamba install -y jupyterlab numpy pandas xarray netcdf4 matplotlib cartopy geopandas shapely pyproj dask ipywidgets
```
Do not install Cartopy with pip; install it via mamba/conda only.

#### 2.2.6 Launch JupyterLab and pick the kernel
From the root of this repository:

```bash
jupyter lab
```
In JupyterLab:

Open any notebook.

Go to “Kernel → Change Kernel”.

Select “Python (climate-env)”.

This ensures notebooks use the environment you just created.

#### 2.2.7 Quick sanity check
In any notebook, run:

python
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

print("Environment setup successful!")
If this cell runs without errors, your local environment is ready.

#### 2.2.8 Optional: export the environment
To share your exact environment with others:

```bash
conda env export > environment.yml
```
Others can then recreate it with:

```bash
conda env create -f environment.yml
```
#### 2.2.9 Troubleshooting
conda: command not found

Restart the terminal.

Ensure Miniforge added itself to your PATH at install time.

Cartopy installation fails

Use mamba, not pip.

Confirm you are using Python 3.10.

Kernel not visible in Jupyter

Activate climate-env before starting jupyter lab.

Restart JupyterLab after creating the environment.

## 3. Notebook overview
The notebooks are organised by function; you can often open an individual notebook directly once you have the appropriate dataset available.

Order	Notebook	Main role
01	01_accessing_data.ipynb	Access NARCliM 2.0 data via NCI THREDDS / OPeNDAP.
02	02_regridding.ipynb	Regrid from native WRF/NARCliM grid to regular lat–lon.
03	03_wa_nrm_region.ipynb	WA NRM region masks and regional time series.
04	04_mapping_and_visualisation.ipynb	Static maps and time series for WA subsets.
05	05_basic_climate_analysis.ipynb	Seasonal means, anomalies, GIS-ready exports.
06	06_interactive_analysis.ipynb	Interactive, widget-based exploration.
A typical end-to-end sequence is:

Access data (01).

Regrid if required (02).

Build WA / NRM region summaries on the native grid (03).

Map and visually inspect fields (04).

Compute climate statistics and export products (05).

Explore results interactively (06).

## 4. Notebook details
### 4.1 01_accessing_data.ipynb — Data access
Goal
Provide a standard pattern for connecting to NARCliM 2.0 data on NCI’s THREDDS server using OPeNDAP, without downloading full NetCDF files.

Key capabilities

Build and document OPeNDAP URLs for specific NARCliM 2.0 experiments, models, and variables (e.g. tasmax).

Open remote NetCDF datasets with xarray.open_dataset, using appropriate decode and caching options.

Inspect dataset structure: dimensions, coordinates, attributes, and variable metadata.

Select variables and time slices for use in later notebooks.

When to use

First contact with a new dataset or scenario.

Whenever you need to confirm variable names, units, and time coverage before analysis.

### 4.2 02_regridding.ipynb — Spatial regridding
Goal
Convert native NARCliM / WRF grids to regular latitude–longitude grids suitable for mapping, multi-model comparison, or GIS workflows.

Key capabilities

Use xesmf to construct conservative or bilinear regridders between the native grid and a user-defined target grid.

Handle rotated pole grids by using the 2‑D lat/lon fields as the source grid.

Save regridded output to NetCDF for reuse in 03–05, or for external GIS applications.

When to use

Before overlaying with other gridded datasets.

Before exporting rasters for GIS or web mapping services.

When you want identical grids across scenarios or models.

### 4.3 03_wa_nrm_region.ipynb — WA NRM regions
Goal
Produce region-based summaries using official WA NRM region boundaries, working on the native WRF grid.

Key capabilities

Load the official “NRM Regions 2023” shapefile and filter to Western Australia.

Construct a GeoDataFrame of WRF grid points, storing native indices rlat, rlon and geometries (via shared helpers in climate_utils.py).

Spatially join grid points to NRM polygons so that each grid cell is tagged with its NRM region.

For each region, build boolean masks on the native grid and compute regional time series (e.g. tasmax converted from Kelvin to Celsius).

Plot multi-region time series and build summary tables of annual or monthly mean tasmax per region.

Typical outputs

Seven monthly time series (one per WA NRM region) for tasmax or other variables.

A table of annual mean temperature per region.

A region-labelled point table that can be reused by other notebooks.

When to use

When stakeholders need results summarised at NRM-region scale rather than grid-cell scale.

Prior to building dashboards or tables comparing regions.

### 4.4 04_mapping_and_visualisation.ipynb — Mapping
Goal
Create publication-ready maps and time series for Western Australia, handling the rotated NARCliM grid correctly.

Key capabilities

Subset to a WA bounding box using the 2‑D lat/lon fields and a reusable WA mask (e.g. mask_wa_bbox).

Convert tasmax to degrees Celsius for more intuitive interpretation.

Use Cartopy to plot with pcolormesh(lon, lat, data, transform=ccrs.PlateCarree()), ensuring grid points appear in their true geographic locations.

Overlay coastlines and other Cartopy features for context, and adjust colour scales and labels for clarity.

Typical outputs

Single time-slice maps (e.g. monthly tasmax over WA).

Quick-look plots to confirm that subsetting and regridding worked as expected.

When to use

For rapid visual QC of new datasets or scenarios.

When you need static images for reports, presentations, or documentation.

### 4.5 05_basic_climate_analysis.ipynb — Climate statistics
Goal
Implement core climate analysis steps such as seasonal means, anomalies, and exports suitable for GIS and further analysis.

Key capabilities

Subset tasmax to a WA bounding box using the same mask pattern as 04.

Convert to Celsius for all downstream statistics.

Compute seasonal means (DJF, MAM, JJA, SON) and an annual mean for a chosen year.

Calculate seasonal anomalies relative to the annual mean and generate colour-coded anomaly maps.

Prepare and export regridded products (NetCDF, and optionally COGs) for use in GIS and external tools.

Generate a Perth point time series (or other locations) extracted from the gridded fields.

Typical outputs

Seasonal mean and anomaly fields over WA on a consistent grid.

Location-specific time series for local analysis.

Files ready for QGIS/ArcGIS or web-map publication.

When to use

For “basic climate products” like seasonal maps, anomalies, and simple indicators.

When preparing data for decision-support tools and communication products.

### 4.6 06_interactive_analysis.ipynb — Interactive exploration
Goal
Provide interactive controls (widgets) to explore variables, regions, and time periods in a dashboard-like manner within Jupyter.

Typical capabilities

Dropdowns or sliders for choosing variable, season, or scenario.

Interactive plots that update in response to user selections.

Quick comparison of regions or seasons without editing code.

When to use

For workshops, training, and stakeholder sessions where interactivity helps understanding.

For exploratory analysis when you want rapid visual feedback.

If widgets appear non-responsive, restart the kernel, ensure widget extensions are enabled, and re-run the notebook from the top.

## 5. Recommended workflows
Workflow A — Rapid visual inspection
Use this when you want to see what a dataset “looks like” before investing in heavy processing.

Open 01_accessing_data.ipynb and load the target tasmax (or other variable) from NARCliM 2.0 via THREDDS.

Optionally subset in time to a few months.

Open 04_mapping_and_visualisation.ipynb.

Use the WA mask, convert to Celsius, and plot one or more time steps.

Workflow B — WA NRM regional climate summary
Use this when the end product is a table or set of plots per WA NRM region.

Access the relevant dataset in 01_accessing_data.ipynb.

If needed, regrid in 02_regridding.ipynb (for consistent grids across models).

Build NRM region masks and regional time series in 03_wa_nrm_region.ipynb.

Summarise and visualise results in 03 (time-series plots, summary tables) or in 05 for seasonal/statistical context.

Workflow C — Seasonal anomalies and GIS export
Use this when preparing map products or GIS layers.

Open 01_accessing_data.ipynb and select a single year and scenario.

Optionally process/regrid in 02_regridding.ipynb.

In 05_basic_climate_analysis.ipynb, subset to WA, convert to Celsius, compute seasonal means and anomalies, and export to NetCDF/COG.

Import exported files into GIS or other visualisation tools as needed.

Workflow D — Interactive exploration
Use this when you want a lightweight dashboard.

Prepare baseline datasets with 01–05.

Open 06_interactive_analysis.ipynb.

Use widgets to choose region, variable, period, and inspect results dynamically.

## 6. Data handling and persistence
Raw NARCliM 2.0 data are not stored in this repository; they are accessed directly from NCI’s THREDDS server via OPeNDAP.

Notebook outputs (plots, NetCDF, COGs) are generated on demand and can be saved locally or committed back to the repository.

Hosted environments (Binder, etc.) do not retain state after shutdown; always download or commit outputs you need to keep.

## 7. Best practices
To keep workflows efficient and robust:

Inspect datasets (dimensions, variables, units) in 01 before running heavy computations.

Subset spatially (WA mask) and temporally (e.g. one year) before regridding or aggregating.

Use shared helpers in climate_utils.py (e.g. kelvin_to_celsius, mask_wa_bbox, build_grid_points) to avoid duplicated logic.

Regrid before mapping if the grid is rotated or irregular and you need GIS compatibility (02, 05).

Prefer regional aggregation (03) when communicating with planners or managers.

Ensure figures include variable name, units, time period, and scenario in titles or labels.

Record key parameter choices (bounding boxes, seasons, regridding method) in markdown cells for reproducibility.

## 8. Common issues and fixes
Blank maps

Check that the WA mask did not remove all cells (inspect mask.sum()).

Confirm lat/lon variable names and that you are using the correct Cartopy transform (PlateCarree).

Very slow notebooks

Reduce temporal extent (fewer years or months).

Apply the WA bounding box before regridding or aggregating.

Run locally rather than via Binder for large jobs.

Broken or non-responsive widgets (06)

Restart kernel, run all cells.

Ensure ipywidgets/extensions are installed and enabled in your Jupyter environment.

## 9. Intended use
This collection is intended for:

Climate data exploration and “first pass” assessments over Western Australia.

Regional climate summaries at WA NRM region scale.

Prototyping methods and workflows that may later be scaled up or automated.

Generating reproducible examples for reports, training, and collaboration.

The design emphasises:

Modularity — each notebook can be run on its own when appropriate inputs exist.

Transparency — all processing steps are visible, inspectable, and version-controlled.

Extensibility — new datasets, time periods, and regions can be added with minimal structural change.