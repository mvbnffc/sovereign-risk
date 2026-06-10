# Sovereign Climate-Risk Model for Thailand

This repository contains the analysis code for an academic paper modelling how
river-flood risk and flood-protection adaptation can affect Thailand's
macroeconomy and sovereign credit rating. The workflow links:

1. physical flood hazard,
2. exposed assets and sectoral activity,
3. basin-level flood losses,
4. macroeconomic shocks through DIGNAD, and
5. sovereign credit-rating impacts.

The code is organised as a small Python package (`sovereign/`) plus notebooks
that run the research pipeline from data preparation through paper figures.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `sovereign/` | Reusable Python functions for flood-risk processing, DIGNAD coupling, and DIGNAD pre-computation. |
| `notebooks/preparation/` | Final raw-data preparation notebooks for exposure, flood protection, future flood shifts, and discharge dependence. |
| `notebooks/pre_sim/` | Final expensive pre-simulation notebooks, especially flood overlays, basin aggregation, and DIGNAD pre-computation. |
| `notebooks/paper_analysis/` | Final paper simulations, validation, and figures. |
| `notebooks/0.*` to `notebooks/4_*` | Earlier/development numbered workflow notebooks. |
| `inputs/` | Required raw and prepared input data. Ignored by Git; publish separately as a data bundle. |
| `outputs/` | Generated outputs and reproducibility caches. Ignored by Git; selected outputs should be archived with the data bundle. |
| `DIGNAD/` | Local DIGNAD toolkit installation. Ignored by Git; see DIGNAD notes below. |

## Quick Start

Create a Python environment. Conda/mamba is recommended because the workflow
uses geospatial packages (`rasterio`, `geopandas`, `cfgrib`) that depend on GDAL
and ECMWF GRIB tooling.

```bash
conda env create -f environment.yml
conda activate sovereign-risk
pip install -e .
```

If you prefer pip, the editable install also declares the Python dependencies:

```bash
pip install -e .
```

The full workflow also requires MATLAB on `PATH` and the DIGNAD toolkit. See
the DIGNAD setup notes below.

```text
DIGNAD/DIGNAD_Toolkit/
```

## Reproducibility

The data and output bundles are hosted separately from the code repository.
Replace the placeholder links below with the final data repository URLs.

### Option A: Full Model Reproduction

Use this option to rebuild the full analysis from source inputs.

Required bundle:

- `inputs/` folder: `[placeholder: input data repository link]`

Steps:

1. Download and unpack `inputs/` into the repository root.
2. Install the Python package and configure DIGNAD as described below.
3. Start Jupyter from the repository root:

   ```bash
   jupyter notebook
   ```

4. Run all notebooks in the order shown in the runtime table.

### Option B: Paper Simulation Reproduction

Use this option to reproduce the final paper simulations without rerunning all
preparation and pre-simulation notebooks.

Required bundles:

- `inputs/` folder: `[placeholder: input data repository link]`
- selected `outputs/` folders needed by the paper simulation notebooks:
  `[placeholder: selected output repository link]`

Steps:

1. Download and unpack the required `inputs/` and selected `outputs/` folders
   into the repository root.
2. Install the Python package and configure DIGNAD as described below.
3. Run:
   - `notebooks/paper_analysis/full_simulation.ipynb`
   - `notebooks/paper_analysis/validation_2011_floods.ipynb`
   - `notebooks/paper_analysis/figures.ipynb`

### Option C: Figure Reproduction

Use this option to reproduce only the paper figures from archived model outputs.

Required bundles:

- `inputs/` folder: `[placeholder: input data repository link]`
- all `outputs/` folders: `[placeholder: output data repository link]`

Steps:

1. Download and unpack the required `inputs/` and `outputs/` folders into the
   repository root.
2. Install the Python package.
3. Run `notebooks/paper_analysis/figures.ipynb`.

## Notebook Runtime

Approximate runtimes are shown for the final workflow notebooks.

| Notebook | Runtime |
| --- | --- |
| `notebooks/preparation/exposure.ipynb` | 10 mins |
| `notebooks/preparation/flood_protection.ipynb` | 2 mins |
| `notebooks/preparation/copula_flood_dependence.ipynb` | 5 mins |
| `notebooks/preparation/future_flood.ipynb` | 3 hours |
| `notebooks/pre_sim/1_flood_risk_overlay.ipynb` | 1 hour |
| `notebooks/pre_sim/2_flood_risk_basin_sum.ipynb` | 10 mins |
| `notebooks/pre_sim/3_dignad_pre_simulation.ipynb` | 28 hours |
| `notebooks/paper_analysis/full_simulation.ipynb` | 3 hours |
| `notebooks/paper_analysis/validation_2011_floods.ipynb` | 20 mins |
| `notebooks/paper_analysis/figures.ipynb` | 5 mins |

## Important External Dependency: DIGNAD

`sovereign/macroeconomic.py::run_DIGNAD()` modifies DIGNAD workbook inputs,
calls MATLAB, and reads the resulting Excel output. A successful run requires:

- MATLAB available on the system `PATH`;
- DIGNAD downloaded from the IMF DIGNAD page:
  <https://climatedata.imf.org/pages/dignad>;
- the DIGNAD toolkit copied into this repository under `DIGNAD/`;
- the Thailand-calibrated `input_DIG-ND.xlsx` workbook copied into the DIGNAD
  toolkit folder, replacing the default workbook.

Download `DIGNAD.zip` from the IMF website, unzip it, and copy the extracted
`DIGNAD` folder into the repository root. The intended structure is:

```text
sovereign-risk/
  DIGNAD/
    DIGNAD_Toolkit/
      input_DIG-ND.xlsx
      simulate.m
      ...
```

The key model files, including `simulate.m`, should be inside
`DIGNAD/DIGNAD_Toolkit/`. If the IMF zip extracts to a slightly different
folder layout, adjust the copied folder so that the Python code can find:

```text
DIGNAD/DIGNAD_Toolkit/input_DIG-ND.xlsx
DIGNAD/DIGNAD_Toolkit/simulate.m
```

The default IMF `input_DIG-ND.xlsx` should be replaced with the pre-prepared
Thailand-calibrated workbook supplied with this project. That workbook is set up
to be modified by the Python scripts before each DIGNAD run.

Only redistribute the DIGNAD toolkit if its licence permits redistribution.
Otherwise, point users to the official DIGNAD source.
