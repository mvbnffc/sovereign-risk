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
| `notebooks/preparation/` | Raw-data preparation notebooks for exposure, flood protection, future flood shifts, and discharge dependence. |
| `notebooks/pre_sim/` | Expensive pre-simulation notebooks, especially flood overlays and DIGNAD pre-computation. |
| `notebooks/0.*` to `notebooks/4_*` | Main numbered modelling workflow. |
| `notebooks/paper_analysis/` | Final paper simulations, validation, and figures. |
| `inputs/` | Required raw and prepared input data. Ignored by Git; publish separately as a data bundle. |
| `outputs/` | Generated outputs and reproducibility caches. Ignored by Git; selected outputs should be archived with the data bundle. |
| `DIGNAD/` | Local DIGNAD toolkit installation. Ignored by Git; see DIGNAD notes below. |
| `docs/` | Reproducibility, data-publication, and repository-map documentation. |

For a fuller module and notebook map, see
[`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md).

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

The full workflow also requires MATLAB on `PATH` and the DIGNAD toolkit at:

```text
DIGNAD/DIGNAD_Toolkit/DIGNAD_Toolkit/
```

## Reproducing the Analysis

1. Download and unpack the data bundle into the repository root so that
   `inputs/`, selected `outputs/`, and, where redistributable, `DIGNAD/` match
   the structure in [`docs/DATA.md`](docs/DATA.md).
   You can check the unpacked layout with:

   ```bash
   python scripts/check_reproducibility_inputs.py
   ```

2. Start Jupyter from the repository root:

   ```bash
   jupyter notebook
   ```

3. Run the notebooks in the order described in
   [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

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
- add the data DOI to this README, `docs/DATA.md`, and the paper's data
  availability statement after deposit.

See [`docs/DATA.md`](docs/DATA.md) for a concrete manifest and hosting options.

## Important External Dependency: DIGNAD

`sovereign/macroeconomic.py::run_DIGNAD()` modifies DIGNAD workbook inputs,
calls MATLAB, and reads the resulting Excel output. A successful run requires:

- MATLAB available on the system `PATH`;
- DIGNAD installed at `DIGNAD/DIGNAD_Toolkit/DIGNAD_Toolkit/`;
- `input_DIG-ND.xlsx` inside that directory;
- `simulate.m` inside that directory.

Only redistribute the DIGNAD toolkit if its licence permits redistribution.
Otherwise, archive a short setup note and point users to the official DIGNAD
source.

## Publication-Readiness Checklist

- [ ] Publish the code repository.
- [ ] Publish the data bundle and add the DOI in `docs/DATA.md`.
- [ ] Clean or clear notebook outputs that contain local absolute paths.
- [ ] Move paper-facing results into `notebooks/paper_analysis/`.
- [ ] Confirm the DIGNAD redistribution/licensing position.
- [ ] Add the final paper citation and DOI once available.
