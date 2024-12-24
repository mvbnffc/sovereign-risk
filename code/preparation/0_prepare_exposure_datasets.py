"""
Script Name: 0_prepare_exposure_datasets.py
Description: Prepares the gridded exposure datasets for the flood risk analysis.
Author: Mark Bernhofen
Date: 2024-12-24
"""

import os
import geopandas as gpd
import rasterio
import numpy as np
import logging
from preparation_functions import write_raster, check_directory_exists
from rasterio.features import geometry_mask

def disaggregate_values(gva_data, raster, sector):
    '''
    This function disaggregates GVA values at the sub-national scale (Admin 1) to rasters.
    Disaggregation is weighted by raster values within each sub-national region.
    '''

    # Initialize an empty array for the output raster values
    output_values = np.zeros_like(raster.read(1), dtype=np.float32)
    
    for index, row in gva_data.iterrows():

        # Value for sector?
        total_value = row[sector]

        # Proceed if there's a meaningful total value to disaggregate
        if total_value > 0:
            # Create a mask for the admin area
            geom_mask = geometry_mask([row['geometry']], invert=True, transform=raster.transform, out_shape=raster.shape)
            
            # Calculate the total area of the admin area covered by buildings in the raster
            total_area = raster.read(1)[geom_mask].sum()
                
            if total_area > 0:  # Prevent division by zero
                # Disaggregate the total value across the raster, weighted by building area
                output_values[geom_mask] += (raster.read(1)[geom_mask] / total_area) * total_value
        

    return  output_values

def main():

    # Set directories and paths
    # proj_path = "/Users/markbernhofen/Projects/sovereign-risk"
    proj_path = "/soge-home/users/smit0210/sovereign-risk" # path for linux cluster
    input_data_path = os.path.join(proj_path, "data", "Inputs", "Flood", "Exposure")
    check_directory_exists(input_data_path) # will exit if it doesn't exist
    output_data_path = os.path.join(proj_path, "data", "Outputs", "0_exposure_data")
    check_directory_exists(output_data_path, False) # will create if it doesn't exist
    logging.info("Setting directories and paths")

    # Step 1: Load the subnational economic data
    economic_data = gpd.read_file(os.path.join(input_data_path, "DOSE_V2_THA_rgva.gpkg"))  # Ensure this has 'GID_1' column
    logging.info("Loading admin-level economic data")

    # Step 2: Load the gridded data 
    nres = rasterio.open(os.path.join(input_data_path, 'GHSL_nres_THA_v.tif'))
    res = rasterio.open(os.path.join(input_data_path, 'GHSL_res_THA_v.tif'))
    agr = rasterio.open(os.path.join(input_data_path, 'cropland_binary.tif'))
    logging.info("Loading gridded data")

    # Step 3: Disaggregate GVA values
    agr_dis = disaggregate_values(economic_data, agr, 'ag_grp')
    man_dis = disaggregate_values(economic_data, nres, 'man_grp')
    nr_ser_dis = disaggregate_values(economic_data, nres, 'serv_grp')*0.709 # to account for service sector minus real estate
    res_dis = disaggregate_values(economic_data, res, 'serv_grp')*0.291 # real estate service sector
    ser_dis = res_dis + nr_ser_dis # combine into one service sector layer
    logging.info("Disaggregating economic data across grid")

    # Step 4: Write rasters
    write_raster(os.path.join(output_data_path, 'agr_GVA_THA.tif'), agr, agr_dis)
    write_raster(os.path.join(output_data_path, 'man_GVA_THA.tif'), nres, man_dis)
    write_raster(os.path.join(output_data_path, 'ser_GVA_THA.tif'), nres, ser_dis)
    logging.info("Writing exposure rasters to disk")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()