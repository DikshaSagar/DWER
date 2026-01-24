# Climate Data Analysis

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DikshaSagar/DWER/HEAD)

A modular Jupyter-based workflow for accessing, analysing, and visualising regional climate model data, with support for **remote data access**, **spatial processing**, **regional aggregation**, and **interactive exploration**.

This repository is designed for use both **online via Binder** and **locally** for extended analysis.

---

## Overview

This toolkit provides a structured approach to working with regional climate datasets, including:

- Direct access to remote NetCDF data via THREDDS / OPeNDAP  
- Spatial preparation through subsetting and regridding  
- Regional-scale analysis using WA Natural Resource Management (NRM) boundaries  
- Consistent visualisation workflows  
- Interactive exploration using Jupyter widgets  

Each notebook performs a defined function and can be used independently.

---

## Access options

### Online (recommended for quick access)
Use Binder to run the notebooks in your browser with no local installation.

Click the **Launch Binder** badge above to start.

### Local execution
Recommended for large datasets and computationally intensive workflows.

---

## Repository structure

The repository is organised by **function**, not by dependency.  
Notebooks can be run independently.

Typical workflow order:

1. Overview  
2. Data access  
3. Spatial preparation  
4. Regional aggregation  
5. Visualisation  
6. Climate summaries  
7. Interactive exploration  

---

## Notebook overview

### Overview  
Provides context, scope, and workflow orientation.

### Data access  
Implements standardised access to NARCliM climate data via THREDDS / OPeNDAP services.

### Spatial regridding  
Prepares datasets for consistent spatial analysis and mapping.

### Regional analysis (WA NRM regions)  
Aggregates gridded climate data to regional boundaries for management-scale summaries.

### Plotting and visualisation  
Produces spatial maps and time-series figures using consistent projection handling.

### Basic climate analysis  
Implements core climate statistics and summary routines.

### Interactive analysis  
Adds dynamic controls for exploratory analysis within Jupyter.

---

## Getting started with Binder

1. Open this repository on GitHub.  
2. Click the **Launch Binder** badge.  
3. Wait for the environment to initialise.  
4. JupyterLab will open in your browser.  
5. Select any notebook and execute cells using **Shift + Enter**.

> Binder sessions are temporary.  
> Download outputs or commit changes before closing the session.

---

## Local setup

### Clone the repository
```bash
git clone <repository-url>
cd <repository-folder>

-----

### Create an environment
conda create -n climate-toolkit python=3.10
conda activate climate-toolkit
pip install -r requirements.txt
jupyter lab

### Launch Jupyter
jupyter lab

-------

## Acknowledgement

This repository contains example workflows, analysis notebooks, and supporting utilities developed as part of a Climate Science Initiative (CSI) project. The material has been informed by contributions, feedback, and established best practices from the climate science and data analysis community.

The structure of the notebooks and analytical approaches adopted here draw inspiration from community-led efforts in climate model evaluation and data analysis, including methodologies developed by the Program for Climate Model Diagnosis and Intercomparison (PCMDI) and related model intercomparison initiatives. These influences are acknowledged as guidance rather than direct dependencies.

Regional climate projection data used throughout this repository are sourced from the NARCliM2.0 project and accessed via National Computational Infrastructure (NCI) THREDDS services. Users are encouraged to consult the original data providers and official documentation for authoritative descriptions of the datasets, metadata, and recommended usage.

This repository is intended to serve as an open, evolving resource for demonstration, exploration, and learning, supporting transparent and reproducible climate data analysis workflows.

## Disclaimer

The workflows, analyses, and visualisations provided in this repository are supplied for demonstration and exploratory purposes only. While care has been taken to ensure methodological clarity and internal consistency, the outputs should not be used as the sole basis for decision-making without independent validation and consultation of authoritative data sources.

Any interpretations, conclusions, or errors arising from the use of this repository remain the responsibility of the authors and contributors.
