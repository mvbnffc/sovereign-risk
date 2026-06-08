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
| `notebooks/0.*` to `notebooks/4_*` | Earlier/development numbered workflow notebooks; review before deleting because some may still contain useful checks. |
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
DIGNAD/DIGNAD_Toolkit/DIGNAD_Toolkit/
```

## Reproducing the Analysis

1. Download and unpack the data bundle into the repository root so that
   `inputs/`, selected `outputs/`, and, where redistributable, `DIGNAD/` match
   the expected project structure.
2. Start Jupyter from the repository root:

   ```bash
   jupyter notebook
   ```

3. Run the notebooks in the analysis order below.

## Working Analysis Map

Use this as the working checklist while rerunning and cleaning the repository.

| Stage | Keep / run | Purpose | Decision after rerun |
| --- | --- | --- | --- |
| 0a | `notebooks/preparation/exposure.ipynb` | Prepare gridded exposure and economic value layers. | Keep as canonical if it regenerates the exposure outputs used downstream. |
| 0b | `notebooks/preparation/flood_protection.ipynb` | Process FLOPROS protection and adaptation-cost layers. | Keep as canonical if adaptation cost/results are used in the paper. |
| 0c | `notebooks/preparation/copula_flood_dependence.ipynb` | Fit inter-basin GloFAS discharge dependence and create copula samples. | Keep as canonical copula/dependence notebook. |
| 0d | `notebooks/preparation/future_flood.ipynb` | Process ISIMIP future discharge into basin return-period shifts. | Keep as canonical future-climate notebook. |
| 1 | `notebooks/pre_sim/1_flood_risk_overlay.ipynb` | Overlay flood hazard, vulnerability, and exposure. | Keep as canonical flood-risk overlay notebook. |
| 2 | `notebooks/pre_sim/2_flood_risk_basin_sum.ipynb` | Aggregate risk rasters to basin/admin/sector losses. | Keep as canonical basin aggregation notebook. |
| 3 | `notebooks/pre_sim/3_dignad_pre_simulation.ipynb` | Pre-compute DIGNAD outputs for interpolation. | Keep, but only rerun when DIGNAD parameters/grid changes. |
| 4 | `notebooks/paper_analysis/full_simulation.ipynb` | Final integrated flood-macro-credit simulation. | Treat as the main paper-results notebook. |
| 5 | `notebooks/paper_analysis/validation_2011_floods.ipynb` | Validate flood/macro/credit chain against 2011 Thailand floods. | Keep if validation is in the paper or supplement. |
| 6 | `notebooks/paper_analysis/figures.ipynb` | Produce final paper figures and summary outputs. | Treat as the final figure-generation notebook. |

Likely deletion/archive candidates after the rerun:

- `notebooks/scratch_DIGNAD_presim.ipynb`
- `notebooks/dignad_four_parameter_workflow.ipynb`, if fully superseded by
  `notebooks/pre_sim/3_dignad_pre_simulation.ipynb`
- `notebooks/flood_and_macro_sim.ipynb`, if superseded by
  `notebooks/paper_analysis/full_simulation.ipynb`
- `notebooks/full_model_framework_simulation.ipynb`, if superseded by
  `notebooks/paper_analysis/full_simulation.ipynb`
- `notebooks/paper_figures.ipynb`, if superseded by
  `notebooks/paper_analysis/figures.ipynb`
- `notebooks/0.1_data_prep.ipynb`
- `notebooks/0.2_Copula_Fitting.ipynb`, if superseded by
  `notebooks/preparation/copula_flood_dependence.ipynb`
- `notebooks/0.3_future_flood.ipynb`, if superseded by
  `notebooks/preparation/future_flood.ipynb`
- `notebooks/1_flood_risk_overlay.ipynb`, if superseded by
  `notebooks/pre_sim/1_flood_risk_overlay.ipynb`
- `notebooks/2_risk_basin_zonal_sum.ipynb`, if superseded by
  `notebooks/pre_sim/2_flood_risk_basin_sum.ipynb`
- `notebooks/3_national_flood_simulation.ipynb`,
  `notebooks/3_national_flood_simulation_future.ipynb`, and
  `notebooks/4_macro_simulation.ipynb`, if superseded by
  `notebooks/paper_analysis/full_simulation.ipynb`

The DIGNAD pre-computation can take many hours because individual MATLAB runs
are executed repeatedly. For paper reproduction, archive the pre-computed DIGNAD
CSV outputs with the data bundle so users can reproduce final results without
rerunning every DIGNAD scenario.

## Data Availability Plan

The current local `inputs/` folder is about 4.6 GB and `outputs/` is about
2.2 GB. These files should not be committed to Git. Recommended release pattern:

- host source code on GitHub or another Git repository;
- host `inputs/` plus required cached outputs on a research-data repository with
  a DOI, preferably Zenodo for this size class;
- add the data DOI to this README and the paper's data availability statement
  after deposit.

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
      DIGNAD_Toolkit/
        input_DIG-ND.xlsx
        simulate.m
        ...
```

The key model files, including `simulate.m`, should be inside the innermost
`DIGNAD_Toolkit` folder. If the IMF zip extracts to a slightly different folder
layout, adjust the copied folder so that the Python code can find:

```text
DIGNAD/DIGNAD_Toolkit/DIGNAD_Toolkit/input_DIG-ND.xlsx
DIGNAD/DIGNAD_Toolkit/DIGNAD_Toolkit/simulate.m
```

The default IMF `input_DIG-ND.xlsx` should be replaced with the pre-prepared
Thailand-calibrated workbook supplied with this project. That workbook is set up
to be modified by the Python scripts before each DIGNAD run.

Only redistribute the DIGNAD toolkit if its licence permits redistribution.
Otherwise, archive a short setup note and point users to the official DIGNAD
source.

## Publication-Readiness Checklist

- [ ] Publish the code repository.
- [ ] Publish the data bundle and add the DOI in this README.
- [ ] Clean or clear notebook outputs that contain local absolute paths.
- [ ] Move paper-facing results into `notebooks/paper_analysis/`.
- [ ] Confirm the DIGNAD redistribution/licensing position.
- [ ] Add the final paper citation and DOI once available.
