"""Check whether the expected reproducibility data layout is present.

Run from the repository root after unpacking the data bundle:

    python scripts/check_reproducibility_inputs.py
"""

from __future__ import annotations

from pathlib import Path


REQUIRED_PATHS = [
    "inputs/boundaries/admin/gadm36_THA_1.shp",
    "inputs/boundaries/basins/BA_THA_lev06.shp",
    "inputs/credit_risk/economic.csv",
    "inputs/credit_risk/PD_ratings.csv",
    "inputs/credit_risk/T3.csv",
    "inputs/exposure/capstock/giri_capstock.csv",
    "inputs/exposure/gva/DOSE_V2p11_THA_rgva.csv",
    "inputs/flood/dependence/basin_outlets_match.csv",
    "inputs/flood/dependence/glofas/uparea_glofas_v4_0.nc",
    "inputs/flood/maps/THA_jrc-flood_RP100.tif",
    "inputs/flood/protection/flopros-adm1-map.csv",
    "inputs/flood/protection/flopros-THA.shp",
    "inputs/flood/rivers/hydroRIVERS_v10_THA.shp",
    "inputs/flood/vulnerability/jrc_depth_damage.csv",
    "inputs/macro/THA_2022_calibration_final.csv",
]

RECOMMENDED_CACHE_PATHS = [
    "outputs/flood/dependence/copulas/copula_random_numbers.gzip",
    "outputs/flood/future/basin_rp_shifts.csv",
    "outputs/flood/risk/basins/risk_basins.csv",
    "outputs/results/full_model_simulation.csv",
]

DIGNAD_PATHS = [
    "DIGNAD/DIGNAD_Toolkit/DIGNAD_Toolkit/input_DIG-ND.xlsx",
    "DIGNAD/DIGNAD_Toolkit/DIGNAD_Toolkit/simulate.m",
]


def report(paths: list[str], root: Path, label: str) -> int:
    print(f"\n{label}")
    missing = 0
    for rel_path in paths:
        path = root / rel_path
        if path.exists():
            print(f"  OK      {rel_path}")
        else:
            print(f"  MISSING {rel_path}")
            missing += 1
    return missing


def main() -> int:
    root = Path.cwd()
    print(f"Checking reproducibility inputs under: {root}")

    missing_required = report(REQUIRED_PATHS, root, "Required inputs")
    missing_cache = report(RECOMMENDED_CACHE_PATHS, root, "Recommended cached outputs")
    missing_dignad = report(DIGNAD_PATHS, root, "DIGNAD files")

    print("\nSummary")
    print(f"  Required inputs missing: {missing_required}")
    print(f"  Recommended caches missing: {missing_cache}")
    print(f"  DIGNAD files missing: {missing_dignad}")

    if missing_required:
        print("\nRequired inputs are missing; restore the data bundle before rerunning.")
        return 1

    if missing_dignad:
        print("\nDIGNAD is incomplete; macroeconomic reruns will fail until it is installed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
