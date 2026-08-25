#file1

import rasterio
import numpy as np
import pandas as pd
import os
from rasterio.transform import rowcol

# Define directories for all climate variables
tmax_dir = "wc2.1_30s_tmax"
tmin_dir = "wc2.1_30s_tmin"
tavg_dir = "wc2.1_30s_tavg"
wind_dir = "wc2.1_30s_wind"
prec_dir = "wc2.1_30s_prec"
vapr_dir = "wc2.1_30s_vapr"
srad_dir = "wc2.1_30s_srad"
elev_file = "wc2.1_30s_elev.tif"  # Single file for elevation

# List of all climate variables to process
variables = {
    'tmax': tmax_dir,
    'tmin': tmin_dir,
    'tavg': tavg_dir,
    'wind': wind_dir,
    'prec': prec_dir,
    'vapr': vapr_dir,
    'srad': srad_dir,
    'elev': None  # Elevation is a single file, not a directory
}

# Generate file paths for each month (1-12) for all variables
file_paths = {}
for var, dir_name in variables.items():
    if var == 'elev':  # Special handling for elevation (single file)
        file_paths[var] = [elev_file]
    else:  # Monthly files for climate variables
        file_paths[var] = [os.path.join(dir_name, f"wc2.1_30s_{var}_{month:02d}.tif") for month in range(1, 13)]

# Load coordinates from the combined_data file
combined_data = pd.read_csv("VGUniversalDatasetCSV.csv")

# Ensure we have longitude and latitude columns
if 'Longitude' not in combined_data.columns or 'Latitude' not in combined_data.columns:
    print("Error: combined_data file must contain 'Longitude' and 'Latitude' columns")
    exit(1)

# Create new columns for all climate variables and elevation
for var in variables.keys():
    combined_data[f"{var.capitalize()}"] = np.nan

# Function to extract values from files for a given coordinate
def extract_values(files, lon, lat):
    values = []
    
    for file in files:
        try:
            with rasterio.open(file) as src:
                # Convert geographic coordinates to raster row/col indices
                row_idx, col_idx = rowcol(src.transform, lon, lat)
                
                # Check if the point is within the raster bounds
                if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                    # Read only the single pixel value
                    window = ((row_idx, row_idx+1), (col_idx, col_idx+1))
                    value = src.read(1, window=window)[0][0]
                    
                    # Store the value without scaling
                    if value != src.nodata:
                        values.append(float(value))
        except Exception as e:
            print(f"Error processing {file} for point ({lon}, {lat}): {e}")
    
    return values

# Process each coordinate point
total_points = len(combined_data)
for idx, row in combined_data.iterrows():
    lon, lat = row['Longitude'], row['Latitude']
    
    # Extract values for each climate variable and elevation
    for var in variables.keys():
        values = extract_values(file_paths[var], lon, lat)
        
        # Calculate and store annual averages (for climate variables) or the single value (for elevation)
        if values:
            if var == 'elev':  # Elevation is a single value
                combined_data.at[idx, f"{var.capitalize()}"] = values[0]
            else:  # Climate variables: average over months
                combined_data.at[idx, f"{var.capitalize()}"] = np.mean(values)
    
    # Print progress
    if (idx + 1) % 10 == 0 or idx == total_points - 1:
        print(f"Processed {idx + 1}/{total_points} points ({(idx + 1) / total_points * 100:.1f}%)")

# Save the updated data
combined_data.to_csv("combined_data_with_climate_and_elev.csv", index=False)
print("Results saved to 'combined_data_with_climate_and_elev.csv'")

# Print a summary
print("\nSummary of data retrieval:")
for var in variables.keys():
    valid_count = combined_data[f"{var.capitalize()}"].notna().sum()
    print(f"  - {var.capitalize()}: {valid_count}/{total_points} points ({valid_count/total_points*100:.1f}%)")
