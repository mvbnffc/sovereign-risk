# Script with functions and classes for flood risk processing

import rasterio
import numpy as np
from dataclasses import dataclass
from typing import Optional
import pandas as pd

def vectorized_damage(depth, value, heights, damage_percents):
    '''
    Vectorized damage function
    Apply damage function given a flood depth and exposure value.
    Function also needs as input the damage function heights > damage_percents
    '''
    # Use np.interp for vectorized linear interpolation
    damage_percentage = np.interp(depth, heights, damage_percents)
    return damage_percentage * value

def calculate_risk(flood, building_values, heights, damage_percents):
    '''
    Pass a flood depth array, array of values, and a vulnerability curve
    to calculate risk.
    '''
    exposure = np.where(flood>0, building_values, 0)
    risk = vectorized_damage(flood, exposure, heights, damage_percents)

    return risk

def simple_risk_overlay(flood_path, exposure_path, output_path, damage_function):
    '''
    This function performs a simple risk overlay analysis.
    It takes as input a flood map, an exposure map, and a vulnerability curve.
    It outputs a risk raster
    '''
    # Load the rasters
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)

    # Data info
    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    nodata = flood.nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        i = 0
        for ji, window in flood.block_windows(1):
            i += 1

            affine = rasterio.windows.transform(window, flood.transform)
            height, width = rasterio.windows.shape(window)
            bbox = rasterio.windows.bounds(window, flood.transform)

            profile.update({
                'height': height,
                'width': width,
                'affine': affine
            })

            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            flood_array = np.where(flood_array>0, flood_array, 0) # remove negative values
            risk = calculate_risk(flood_array, exposure_array, damage_function[0], damage_function[1]) # depths index 0 and prp damage index 1

            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)


def flopros_risk_overlay(flood_path, exposure_path, output_path, mask_path, damage_function):
    '''
    This function performs a risk overlay analysis. Before the risk analysis it masks all urban areas in the exposure dataset
    It takes as input a flood map, an exposure map, an urban area mask map, and a vulnerability curve.
    It outputs a risk raster
    '''
    # Load the rasters
    flood = rasterio.open(flood_path)
    exposure = rasterio.open(exposure_path)
    mask = rasterio.open(mask_path)

    # Data info
    profile = flood.meta.copy()
    profile.update(dtype=rasterio.float32, compress='lzw', nodata=0)
    nodata = flood.nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        i = 0
        for ji, window in flood.block_windows(1):
            i += 1

            affine = rasterio.windows.transform(window, flood.transform)
            height, width = rasterio.windows.shape(window)
            bbox = rasterio.windows.bounds(window, flood.transform)

            profile.update({
                'height': height,
                'width': width,
                'affine': affine
            })

            flood_array = flood.read(1, window=window)
            exposure_array = exposure.read(1, window=window)
            mask_array = mask.read(1, window=window)
            exposure_array = np.where(mask_array==1, 0, exposure_array) # wherever the urban mask equals 1, set to zero in exposure dataset
            flood_array = np.where(flood_array>0, flood_array, 0) # remove negative values
            risk = calculate_risk(flood_array, exposure_array, damage_function[0], damage_function[1]) # depths index 0 and prp damage index 1

            dst.write(risk.astype(rasterio.float32), window=window, indexes=1)


@dataclass
class BasinComponent:
    admin_id: str              # GID_1 or similar
    aeps: np.ndarray            # annual exceedance probabilities
    sector: str                # sector name for flood loss curve
    baseline_losses: np.ndarray # baseline flood losses
    adapted_losses: np.ndarray # adapted flood losses (urban areas masked)
    protection_aep: float       # New_Pr_L or Pr_L
    meta: dict = None

    def baseline_loss_at(self, aep_event: float) -> float:
        return float(np.interp(aep_event, self.aeps, self.baseline_losses))

    def adapted_loss_at(self, aep_event: float) -> float:
        return float(np.interp(aep_event, self.aeps, self.adapted_losses))

    def protected_loss(self, aep_event: float) -> float:
        """Baseline scenario: apply baseline protection only."""
        if aep_event > self.protection_aep:  # protected
            return 0.0
        else:
            return self.baseline_loss_at(aep_event)

    def adapted_loss(self, aep_event:float, adapted_protection_aep: float) -> float:
        """
        Adaptation scenario combining:
        - baseline protection_aep (p_base)
        - stronger adapted_protection_aep (p_adapt)
        """
        p_base = self.protection_aep
        p_adapt = adapted_protection_aep

        # Safety check
        if p_adapt > p_base:
            p_adapt=p_base

        if aep_event > p_base:
            # Baseline protection - no risk
            return 0.0
        elif p_adapt < aep_event < p_base:
            # Above baseline protection but below adapted protection - sample adapted curve
            return self.adapted_loss_at(aep_event)
        else:
            # Above both baseline and adapted protection - sample baseline curve
            return self.baseline_loss_at(aep_event)
        
def build_basin_curves(df: pd.DataFrame):
    basin_dict = {}

    # group by basin, admin, AND sector
    for (basin_id, admin_id, sector), g in df.groupby(["HB_L6", "GID_1", "Sector"]):
        aeps = g["AEP"].to_numpy()
        losses = g["damages"].to_numpy()
        adapted_losses = g["adapted_damages"].to_numpy()

        # Add bankfull 2-year point
        aeps = np.concatenate(([0.5], aeps))
        losses = np.concatenate(([0.0], losses))
        adapted_losses = np.concatenate(([0.0], adapted_losses))

        # sort by AEP ascending for np.interp
        order = np.argsort(aeps)
        aeps = aeps[order]
        losses = losses[order]
        adapted_losses = adapted_losses[order]

        prot = g["Pr_L_AEP"].iloc[0]

        component = BasinComponent(
            admin_id=admin_id,
            sector=sector,
            aeps=aeps,
            baseline_losses=losses,
            adapted_losses=adapted_losses,
            protection_aep=prot,
        )

        basin_dict.setdefault(basin_id, []).append(component)

    return {
        basin_id: BasinLossCurve(basin_id=basin_id, components=components)
        for basin_id, components in basin_dict.items()
    }

@dataclass
class BasinLossCurve:
    basin_id: int
    components: list[BasinComponent]

    def loss_at_event_aep(
        self,
        aep_event,
        scenario: str = "baseline",
        adapted_protection_aep: Optional[float] = None,
    ) -> float:

        if scenario == "baseline" or adapted_protection_aep is None:
            return sum(c.protected_loss(aep_event) for c in self.components)
        elif scenario == "adaptation":
            return sum(c.adapted_loss(aep_event, adapted_protection_aep) for c in self.components)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")


@dataclass
class BasinLossCurve:
    basin_id: int
    components: list[BasinComponent]

    def loss_at_event_aep(
        self,
        aep_event: float,
        scenario: str = "baseline",
        adapted_protection_aep: Optional[float] = None,
        sector: Optional[float] = None,
    ) -> float:
        # filter components if sector is specified
        comps = (
            [c for c in self.components if c.sector == sector]
            if sector is not None else
            self.components
        )

        if scenario == "baseline" or adapted_protection_aep is None:
            return sum(c.protected_loss(aep_event) for c in comps)
        elif scenario == "adaptation":
            return sum(c.adapted_loss(aep_event, adapted_protection_aep) for c in comps)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        

@dataclass
class BasinClimateShift:
    basin_id: int
    ssp: str
    epoch: str
    baseline_rps: np.ndarray   # e.g. [10, 25, 50, 75 100, 200, 500]
    future_rps: np.ndarray     # e.g. [8, 18, 35, 62, 70, 150, 400]

    def adjust_aeps(self, aeps: np.ndarray) -> np.ndarray:
        """
        Take baseline AEPs (1/RP) and return the corresponding
        future AEPs via interpolation on RP-space.
        """
        baseline_rps = 1.0 / aeps
        # interpolate the future RP for each baseline RP
        rp_future = np.interp(
            baseline_rps,
            self.baseline_rps,
            self.future_rps,
            left=self.future_rps[0],
            right=self.future_rps[-1],
        )
        return 1.0 / rp_future
    
def make_uncertainty_curve(values, sds, k=1):
    """Return low and high curves using ±k standard deviations with 0–1 bounds."""
    values = np.array(values)
    sds = np.array(sds)

    low = np.clip(values - k * sds, 0, 1)
    high = np.clip(values + k * sds, 0, 1)

    return low.tolist(), high.tolist()
