**User Guide**

Climate Data Analysis Toolkit
-------------------------------------------------------------------------------------------------------------------
1. Overview

This toolkit provides a structured workflow for accessing, processing, analysing, and visualising regional climate model data using Jupyter Notebooks.

It is designed to support:

Reproducible climate data analysis

Rapid visual inspection of large datasets

Regional-scale climate summaries

Interactive exploration of climate variables

The toolkit is modular. Each notebook performs a defined function and can be used independently.

2. Access modes

The toolkit can be used in two ways:

2.1 Online via Binder

Provides immediate access without local installation.
Suitable for demonstration, exploration, and lightweight analysis.

2.2 Local execution

Recommended for extended workflows, large datasets, and computationally intensive processing.

3. Getting started with Binder

Open the GitHub repository containing this toolkit.

Click the Launch Binder badge.

Wait for the environment to initialise.

JupyterLab will open in the browser.

Select any notebook from the file panel and execute cells using Shift + Enter.

Binder sessions are temporary. Any changes should be downloaded or committed back to the repository before closing the session.

4. Local setup (optional)
4.1 Clone the repository
git clone <repository-url>
cd <repository-folder>

4.2 Create an environment
conda create -n climate-toolkit python=3.10
conda activate climate-toolkit
pip install -r requirements.txt

4.3 Launch Jupyter
jupyter lab

5. Toolkit structure

The notebooks are organised by function, not by dependency.
Each notebook addresses a specific stage of the workflow.

Recommended operational sequence:

Overview

Data access

Spatial preparation

Regional aggregation

Visualisation

Climate summaries

Interactive exploration

6. Functional description of notebooks
6.1 Overview

Purpose:
Provides context for the toolkit, outlines scope, and describes how the workflow is structured.

Supports:

Understanding the analytical scope

Identifying appropriate notebooks for specific tasks

6.2 Data access

Purpose:
Implements a standardised approach for loading NARCliM climate data from THREDDS / OPeNDAP services.

Supports:

Accessing remote NetCDF datasets

Inspecting dataset structure and metadata

Selecting variables and temporal ranges

6.3 Spatial regridding

Purpose:
Prepares datasets for consistent spatial analysis and visualisation.

Supports:

Converting native model grids to regular latitude–longitude grids

Ensuring cross-dataset compatibility

Preparing data for aggregation and mapping

6.4 Regional analysis (WA NRM regions)

Purpose:
Enables region-based climate analysis using official WA Natural Resource Management boundaries.

Supports:

Overlaying gridded climate data with regional polygons

Producing region-level climate summaries

Generating outputs aligned with planning and management scales

6.5 Plotting and visualisation

Purpose:
Provides tools for producing clear, consistent, publication-ready figures.

Supports:

Spatial mapping

Time-series plotting

Projection handling and styling

Export of figures for reports and presentations

6.6 Basic climate analysis

Purpose:
Implements core climate statistics and summary routines.

Supports:

Calculation of means, minima, and maxima

Identification of temporal patterns

Generation of repeatable analytical outputs

6.7 Interactive analysis

Purpose:
Adds an interactive layer for exploratory data analysis.

Supports:

Dynamic variable and time selection

Rapid visual feedback

Prototype dashboard-style interaction within Jupyter

7. Typical operational workflows
Workflow A — Rapid visual inspection

Open Data access

Load dataset

Open Plotting and visualisation

Generate spatial maps

Workflow B — Regional climate summary

Open Data access

Open Spatial regridding

Open Regional analysis

Open Basic climate analysis

Workflow C — Interactive exploration

Open Interactive analysis

Select variables and time range

Explore patterns dynamically

8. Use of Binder
Suitable use cases

Demonstrating workflows

Exploring datasets

Sharing reproducible examples

Running short analyses

Limitations

Not suitable for large datasets

Not suitable for long-running regridding jobs

Sessions are temporary

For computationally intensive tasks, local execution is recommended.

9. Data handling and persistence

Binder environments do not retain data after shutdown.
Any changes, outputs, or figures should be:

Downloaded locally

Or committed to the GitHub repository

10. Best-practice guidance

Inspect datasets before analysis

Subset spatially and temporally before heavy processing

Apply regridding prior to mapping if grids are irregular

Prefer regional aggregation when producing decision-relevant outputs

Ensure figures include:

variable names

units

time coverage

Record parameter choices for reproducibility

11. Common operational issues
Blank maps

Cause: Coordinate mismatch or projection issue
Resolution:

Confirm latitude and longitude variable names

Verify coordinate reference systems

Ensure data is not empty after subsetting

Slow performance

Cause: High spatial or temporal resolution
Resolution:

Reduce time range

Apply spatial subsetting before regridding

Non-responsive widgets

Cause: Widget extensions not initialised
Resolution:
Restart the kernel and re-run the notebook.

12. Intended use

This toolkit supports:

Climate data exploration

Regional climate assessment

Method prototyping

Development of future analytical pipelines

It is designed to be:

Modular — components can be used independently

Transparent — workflows are visible and inspectable

Extendable — new datasets, regions, and methods can be added with minimal restructuring

13. Closing statement

This toolkit provides a structured, reproducible approach to regional climate data analysis in a Jupyter-based environment.

Whether used for:

Exploratory analysis

Regional climate summaries

Visual communication

Workflow prototyping

it establishes a consistent foundation for climate-focused analytical work.
