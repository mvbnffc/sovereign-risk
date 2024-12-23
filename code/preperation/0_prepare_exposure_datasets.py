import os
import geopandas as gpd
import rasterio
import numpy as np
from preperation_functions import write_raster
from rasterio.features import geometry_mask

def disaggregate_values(gva_data, raster, sector):
    '''
    This function disaggregates building values from the GEM exposure database (https://github.com/gem/global_exposure_model)
    at Admin1 level to gridded urban maps maps from GHSL
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

# Set directories and paths
overall_dir = "D:/projects/sovereign-risk/Thailand/data/exposure/gva_prep"
exposure_dir = os.path.join(overall_dir, "raw_exposure")
economic_dir = os.path.join(overall_dir, "raw_gva")
nres_path = os.path.join(exposure_dir, "GHSL_nres_THA_v.tif")
res_path = os.path.join(exposure_dir, "GHSL_res_THA_v.tif")
agr_path = os.path.join(exposure_dir, "cropland_binary.tif")
agr_o_path = os.path.join(overall_dir, "agr_GVA_THA.tif")
man_o_path = os.path.join(overall_dir, "man_GVA_THA.tif")
ser_o_path = os.path.join(overall_dir, "ser_GVA_THA.tif")

# Step 1: Load your subnational economic data
economic_data = gpd.read_file(os.path.join(economic_dir, "DOSE_V2_THA_rgva.gpkg"))  # Ensure this has 'GID_1' column

# Step 2: Load the raster data 
nres = rasterio.open(nres_path)
res = rasterio.open(res_path)
agr = rasterio.open(agr_path)

# Step 3: Disaggregate GVA values
agr_dis = disaggregate_values(economic_data, agr, 'ag_grp')
man_dis = disaggregate_values(economic_data, nres, 'man_grp')
nr_ser_dis = disaggregate_values(economic_data, nres, 'serv_grp')*0.709 # to account for service sector minus real estate
res_dis = disaggregate_values(economic_data, res, 'serv_grp')*0.291 # real estate service sector
ser_dis = res_dis + nr_ser_dis # combine into one service sector layer

# Write rasters
write_raster(agr_o_path, agr, agr_dis)
write_raster(man_o_path, nres, man_dis)
write_raster(ser_o_path, nres, ser_dis)