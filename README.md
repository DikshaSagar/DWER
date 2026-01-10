[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DikshaSagar/DWER/HEAD)
# Western Australia Climate Analysis

This repository provides a set of reusable Jupyter notebooks for accessing,
analysing, and visualising climate model data hosted on the NCI THREDDS
(OpenDAP) infrastructure.

The notebooks are designed as a lightweight, extensible analysis framework
for rapid exploratory analysis and internal workflows, with minimal
assumptions about the underlying dataset beyond standard CF-compliant
metadata.

While the current implementation uses NARCliM 2.0 outputs, the structure
and methods are intentionally dataset-agnostic and are suitable for direct
reuse with future climate datasets hosted on NCI, including CSI products.

---

## Repository Structure

- `00_Overview.ipynb`  
  Project overview, scope, and design intent.

- `01_Data_Access_and_Inspection.ipynb`  
  Examples of accessing climate data directly via THREDDS/OpenDAP and
  inspecting variables, dimensions, units, and time coverage.

- `02_Generic_Spatial_Analysis.ipynb`  
  Variable-agnostic spatial analysis methods, including spatial means,
  time averaging, and basic mapping.

- `03_Temperature_Analysis.ipynb`  
  Temperature-specific diagnostics using `tas`, `tasmax`, and `tasmin`.

- `04_Precipitation_Analysis.ipynb`  
  Precipitation-specific analysis, including appropriate unit handling
  and aggregation.

- `05_Extremes_and_Indices.ipynb`  
  Example extreme climate diagnostics and threshold-based indices.

- `06_Regional_Aggregation.ipynb`  
  Regional summary statistics (mean, minimum, maximum) for Western Australia,
  intended for stakeholder-facing interpretation.

- `utils/`  
  Reusable helper functions for data access, masking, and plotting.

- `environment.yml`  
  Reproducible Conda environment definition.

---

## Usage

This repository is designed to be executed in a reproducible Python
environment defined in `environment.yml`. All notebooks are intended to
be run within a Jupyter environment with dependencies preconfigured.

The repository is hosted on GitHub and is compatible with hosted Jupyter
execution services such as Binder, enabling users to launch and interact
with the notebooks directly without requiring local installation.

---

## Design Principles

- Direct remote access to climate datasets via THREDDS/OpenDAP
- Clear separation between data access, generic analysis, and
  variable-specific diagnostics
- Readable, example-driven notebooks suitable for technical users
- Reproducibility and portability for long-term organisational reuse
