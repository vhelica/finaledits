# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 22:07:28 2025

@author: Ashley
"""
#file4

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 18:03:11 2025

@author: Ashley
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:41:20 2025

@author: Ashley
"""

# -*- coding: utf-8 -*-
"""
Parallelized CartoPy + Folium Visualization of Viviparity Probability
- Reads real lat/lon points from CSV
- Dynamically calculates bounding box
- Efficiently extracts climate data from rasters using multiprocessing
- Predicts viviparity probability
- Generates interpolated contours for Folium map
"""

import os
import numpy as np
import pandas as pd
import rasterio
import pickle
import matplotlib.pyplot as plt
import geojsoncontour
import folium
import branca
from folium import plugins
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

###########################
# PATHS / MODEL
###########################
MODEL_DIR = "saved_models"
CLIMATE_DIR = "."
CSV_FILE = "combined_data_with_climate_and_elev_hand_mod.csv"

# Climate variables for raster extraction
CLIMATE_VARS = {
    'Elev': "wc2.1_30s_elev",
    'Tmax': "wc2.1_30s_tmax",
    'Tmin': "wc2.1_30s_tmin",
    'Tavg': "wc2.1_30s_tavg",
    'Wind': "wc2.1_30s_wind",
    'Prec': "wc2.1_30s_prec",
    'Vapr': "wc2.1_30s_vapr",
    'Srad': "wc2.1_30s_srad"
}

RESOLUTION = .05 # Changed from 5 to 0.5 as requested
DEBUG_MODE = False  # Set to True to print sample extracted values
MAX_WORKERS = 12  # Changed back to 12 threads for better performance
#relanttioship bewtween industrial waste/ air polution/ and radiation/ in predicting antibiotic reistances. 


###########################
# LOAD REAL DATA & BOUNDING BOX
###########################
def load_real_data(csv_file):
    """Load real latitude/longitude points and calculate bounding box."""
    df_real = pd.read_csv(csv_file)

    if not {'Latitude', 'Longitude'}.issubset(df_real.columns):
        raise ValueError("CSV file must contain 'Latitude' and 'Longitude' columns.")
        
    # Check if we have viviparity data (O/V or 0/1) in the file
    has_viviparity_data = False
    viviparity_col = None
    for col_name in ['Viviparity', 'viviparity', 'Viviparous', 'viviparous', 'ReproMode', 'Parity']:
        if col_name in df_real.columns:
            viviparity_col = col_name
            has_viviparity_data = True
            print(f"✅ Found viviparity data in column '{viviparity_col}'")
            # Print sample values to verify format
            sample_values = df_real[viviparity_col].dropna().unique()
            print(f"  - Sample values: {sample_values[:10]}")
            break
    
    if has_viviparity_data:
        # Identify the format - check if using O/V notation or 0/1 notation
        unique_values = set(str(x).upper() for x in df_real[viviparity_col].dropna().unique())
        print(f"  - Unique reproductive mode values: {unique_values}")
        
        # Create a new column for Viviparous (1) / Oviparous (0) classification
        df_real['IsViviparous'] = df_real[viviparity_col].apply(
            lambda x: 1 if str(x).upper() in ['1', 'TRUE', 'YES', 'V', 'VIVIPAROUS', 'VIVIPARITY'] else 0
        )
        
        v_count = df_real['IsViviparous'].sum()
        o_count = len(df_real) - v_count
        print(f"  - Viviparous (V/1): {v_count} species")
        print(f"  - Oviparous (O/0): {o_count} species")
        
        # Verify the conversion worked correctly
        if 'V' in unique_values or 'O' in unique_values:
            v_in_data = sum(1 for x in df_real[viviparity_col] if str(x).upper() == 'V')
            converted_v = sum(1 for i, x in enumerate(df_real[viviparity_col]) 
                             if str(x).upper() == 'V' and df_real['IsViviparous'].iloc[i] == 1)
            
            print(f"  - Verification: Found {v_in_data} 'V' values, converted {converted_v} to 1")
            
            if v_in_data != converted_v:
                print("⚠️ Warning: Conversion may not be accurate. Please check the data.")
    else:
        print("⚠️ No viviparity data found in CSV. Points will be colored uniformly.")
        df_real['IsViviparous'] = -1  # Unknown

    # Manually setting the bounding box for all of Europe and Asia
    # Manually setting the bounding box for all of Europe and Asia
    min_lat, max_lat = 36, 60  # Extended south from 45 to 40
    min_lon, max_lon = -30, 40  # Extended west from -25 to -30


    print(f"📌 Bounding Box: {min_lon}, {min_lat} to {max_lon}, {max_lat}")
    return df_real, min_lon, min_lat, max_lon, max_lat

def load_model(model_name='LogisticRegression'):
    """Load model and related files from MODEL_DIR."""
    model_files = {
        'NeuralNetwork': 'neural_network_model.pkl',
        'RandomForest': 'random_forest_model.pkl',
        'LogisticRegression': 'logistic_model.pkl',  # Add this line
        'LogisticPoly': 'logistic_poly_model.pkl'  # In case you want to use the polynomial version
    }
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        with open(os.path.join(MODEL_DIR, model_files[model_name]), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError as e:
        print(f"Error: Could not find model files in {MODEL_DIR}: {e}")
        raise
###########################
# GENERATE GRID FOR MODEL
###########################
def make_lon_lat_grid(min_lon, min_lat, max_lon, max_lat, resolution=1.0):
    """Generate a grid of lat/lon points with equal spacing in actual distance."""
    # The issue with unequal spacing is because longitude degrees vary in physical distance
    # based on latitude (they get closer together as you move away from the equator)
    
    # Calculate approximate correction factor for longitude at the mean latitude
    # This ensures grid cells are roughly square in terms of actual distance on Earth
    mean_lat_radians = np.radians((min_lat + max_lat) / 2)
    lon_correction = np.cos(mean_lat_radians)
    
    # Account for the correction in longitude spacing
    lon_resolution = resolution / lon_correction
    
    print(f"Using resolution: {resolution}° latitude, {lon_resolution:.4f}° longitude")
    print(f"(Correction factor: {lon_correction:.4f} at latitude {(min_lat + max_lat)/2:.1f}°)")
    
    # Generate grid
    lons = np.arange(min_lon, max_lon, lon_resolution)
    lats = np.arange(min_lat, max_lat, resolution)
    
    # Create meshgrid with properly spaced points
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Convert to DataFrame
    df_grid = pd.DataFrame({'Longitude': lon_grid.ravel(), 'Latitude': lat_grid.ravel()})
    
    print(f"✅ Created {len(df_grid)} grid points.")
    print(f"   Grid dimensions: {len(lats)} rows × {len(lons)} columns")
    print(f"   Latitude range: {min_lat} to {max_lat} ({len(lats)} points)")
    print(f"   Longitude range: {min_lon} to {max_lon} ({len(lons)} points)")
    
    return df_grid

###########################
# PARALLEL CLIMATE DATA EXTRACTION
###########################
def extract_climate_variable(df_grid, var_name, folder_name):
    """Efficiently extract climate data in batch mode, using multiple workers."""
    print(f"🔄 Extracting {var_name} for all points using {MAX_WORKERS} workers...")

    if var_name == "Elev":  # Special case for elevation (single file)
        file_path = os.path.join(CLIMATE_DIR, f"{folder_name}.tif")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist.")
            df_grid[var_name] = np.nan
            return df_grid
            
        values = process_raster_file(file_path, df_grid, var_name)
        df_grid[var_name] = values
        return df_grid

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for month in range(1, 13):
            file_path = os.path.join(CLIMATE_DIR, folder_name, f"{folder_name}_{month:02d}.tif")
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist.")
                continue
                
            futures.append(executor.submit(process_raster_file, file_path, df_grid, var_name))

        if not futures:
            print(f"Warning: No valid files found for {var_name}")
            df_grid[var_name] = np.nan
            return df_grid
            
        results = [f.result() for f in tqdm(futures, desc=f"Processing {var_name}")]

    if not results:
        df_grid[var_name] = np.nan
    else:
        df_grid[var_name] = np.nanmean(results, axis=0)  # Compute mean across 12 months
        
    print(f"✅ Extracted {var_name} for {len(df_grid)} points.")
    
    # Ensure we're always returning a DataFrame
    if not isinstance(df_grid, pd.DataFrame):
        print(f"Converting {var_name} result back to DataFrame")
        # This should never happen, but just in case
        temp_df = pd.DataFrame({'Longitude': df_grid['Longitude'], 'Latitude': df_grid['Latitude']})
        temp_df[var_name] = df_grid[var_name]
        df_grid = temp_df
        
    return df_grid

def process_raster_file(file_path, df_grid, var_name):
    """Read raster file and extract values for all points."""
    print(f"📂 Opening {file_path}...")
    values = np.full(len(df_grid), np.nan)

    try:
        with rasterio.open(file_path) as src:
            raster_data = src.read(1)
            nodata_value = src.nodata if src.nodata is not None else -3.4e+38

            for i, (lon, lat) in enumerate(zip(df_grid["Longitude"], df_grid["Latitude"])):
                try:
                    # Fixed from rasterio.index to src.index
                    row, col = src.index(lon, lat)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        pixel_value = raster_data[row, col]
                        # Improved NoData handling
                        if pixel_value == nodata_value or pixel_value < -100:
                            pixel_value = np.nan
                        values[i] = pixel_value

                        if DEBUG_MODE and i % 500 == 0:
                            print(f"📊 {var_name} at ({lon}, {lat}): {pixel_value}")
                except IndexError:
                    # Point outside raster bounds
                    if DEBUG_MODE and i % 500 == 0:
                        print(f"⚠️ Point ({lon}, {lat}) outside raster bounds")
                    values[i] = np.nan
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return values

    # Make sure we return just the values array, not a modified df_grid
    # This prevents the bug where df_grid becomes a NumPy array
    return values

###########################
# SAVE ENVIRONMENTAL DATA
###########################
def save_environmental_data(df_grid, output_file="environmental_values.csv"):
    """Save all environmental variables by latitude and longitude to a CSV file."""
    # Get all environmental variables (excluding Latitude, Longitude, and Probability)
    env_vars = [col for col in df_grid.columns if col not in ['Latitude', 'Longitude', 'Probability']]
    
    # Create a clean dataframe with lat, long, and all environmental variables
    df_env = df_grid[['Latitude', 'Longitude'] + env_vars].copy()
    
    # Calculate summary statistics
    summary = {}
    for var in env_vars:
        if df_env[var].notna().any():  # Only calculate if we have valid values
            summary[f"{var}_mean"] = df_env[var].mean()
            summary[f"{var}_median"] = df_env[var].median()
            summary[f"{var}_min"] = df_env[var].min()
            summary[f"{var}_max"] = df_env[var].max()
            summary[f"{var}_std"] = df_env[var].std()
    
    # Save data to CSV
    try:
        df_env.to_csv(output_file, index=False)
        print(f"✅ Environmental data saved to {output_file}")
        
        # Also save summary statistics
        summary_df = pd.DataFrame([summary])
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary statistics saved to {summary_file}")
        
        # Print summary to console
        print("\n📊 Summary of Environmental Variables:")
        for var in env_vars:
            if f"{var}_mean" in summary:
                print(f"{var}: Mean={summary[f'{var}_mean']:.2f}, Min={summary[f'{var}_min']:.2f}, Max={summary[f'{var}_max']:.2f}")
            else:
                print(f"{var}: No valid data")
        
    except Exception as e:
        print(f"❌ Error saving environmental data: {e}")
    
    return df_env

def predict_for_grid(df_grid, model, scaler, feature_list):
    """
    Extracts climate data -> predicts viviparity probability 
    for the specific logistic regression model with Wind:Elev interaction
    """
    import statsmodels.api as sm
    import patsy

    # Ensure df_grid is a DataFrame at the start
    if not isinstance(df_grid, pd.DataFrame):
        print("Warning: Input to predict_for_grid is not a DataFrame. Converting...")
        df_grid = pd.DataFrame(df_grid)
    
    # Extract climate variables
    main_features = ['Tmax', 'Tmin', 'Tavg', 'Wind', 'Prec', 'Vapr', 'Srad', 'Elev']
    
    for var in main_features:
        if var not in CLIMATE_VARS:
            print(f"Warning: Climate variable '{var}' not found in CLIMATE_VARS dictionary. Skipping.")
            df_grid[var] = np.nan
            continue
            
        df_grid = extract_climate_variable(df_grid, var, CLIMATE_VARS[var])
    
    # Save environmental data to CSV before prediction
    save_environmental_data(df_grid)

    # Check if we have any valid data
    valid_rows = df_grid.dropna(subset=main_features).copy()
    if len(valid_rows) == 0:
        print("Warning: No valid data points after climate extraction. Check your raster files.")
        df_grid["Probability"] = np.nan
        return df_grid

    # Create specific interaction term (Wind:Elev)
    valid_rows['Wind:Elev'] = valid_rows['Wind'] * valid_rows['Elev']

    # Prepare the design matrix for prediction
    X_pred_df = valid_rows[main_features]
    
    # Scale the main features
    X_pred_scaled = pd.DataFrame(
        scaler.transform(X_pred_df), 
        columns=main_features, 
        index=X_pred_df.index
    )
    
    # Add the interaction term AFTER scaling
    X_pred_scaled['Wind:Elev'] = valid_rows['Wind'] * valid_rows['Elev']
    
    # Predict probabilities
    y_prob = model.predict(X_pred_scaled)
    
    # Create a copy of the original dataframe to preserve all columns
    df_pred = df_grid.copy()
    df_pred["Probability"] = np.nan
    df_pred.loc[valid_rows.index, "Probability"] = y_prob
    
    print(f"✅ Prediction complete for {len(valid_rows)} points out of {len(df_grid)} total.")
    
    # Save final prediction data with environmental variables
    df_pred.to_csv("prediction_with_environmental_data_logit.csv", index=False)
    print("✅ Complete prediction data saved to prediction_with_environmental_data_logit.csv")
    
    return df_pred

def _create_interaction_matrix(X, feature_names):
    """
    Create a design matrix with all main effects and interaction terms
    
    Parameters:
    X (numpy.ndarray): Scaled feature matrix
    feature_names (list): Names of original features
    
    Returns:
    numpy.ndarray: Design matrix with main effects and interactions
    """
    # First, add a column of 1s for the intercept
    interactions = [np.ones(X.shape[0])]
    
    # Add main effects
    interactions.extend([X[:, i] for i in range(X.shape[1])])
    
    # Add interaction terms
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            interactions.append(X[:, i] * X[:, j])
    
    return np.column_stack(interactions)

###########################
# CREATE INTERPOLATED CONTOURS
###########################
def create_contours(df_pred):
    """Generate interpolated contours from predicted values with minimal smoothing to preserve detail."""
    # Filter out NaN values
    df_filtered = df_pred.dropna(subset=["Probability"])
    
    if len(df_filtered) < 10:
        print("Warning: Not enough valid points for interpolation.")
        # Return empty GeoJSON object (not string)
        return {"type": "FeatureCollection", "features": []}, None
        
    x, y, z = df_filtered["Longitude"], df_filtered["Latitude"], df_filtered["Probability"]

    # Create a higher resolution grid for more detailed interpolation
    grid_size = min(300, max(100, len(df_filtered) // 5))  # Increased resolution
    print(f"Using interpolation grid size: {grid_size}x{grid_size}")
    
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size), 
        np.linspace(y.min(), y.max(), grid_size)
    )
    
    # Use nearest interpolation for all points to avoid over-smoothing
    z_mesh = griddata((x, y), z, (x_mesh, y_mesh), method='nearest')
    
    # Only use cubic interpolation for visual smoothness, but preserve the detailed structure
    # by not replacing too many points
    z_cubic = griddata((x, y), z, (x_mesh, y_mesh), method='cubic')
    
    # Only replace nearest with cubic where cubic is valid and conditions are met
    mask = ~np.isnan(z_cubic)
    # Apply less blending to preserve the original data
    z_mesh[mask] = 0.9 * z_mesh[mask] + 0.1 * z_cubic[mask]
    
    # Apply very minimal smoothing to preserve details
    # Reduced sigma from [3,3] to [1,1] for much less smoothing
    z_mesh = ndimage.gaussian_filter(z_mesh, sigma=[.5, .5], mode='nearest')

    # Create more contour levels for finer detail
    levels = 15  # Increased from 10 to 15
    
    # Use a consistent colormap - "YlGnBu" for both the contours and legend
    cmap = plt.cm.YlGnBu
    
    # Create contours
    plt.figure(figsize=(1, 1))  # Small figure to minimize memory usage
    contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels=levels, cmap=cmap)
    try:
        geojson = geojsoncontour.contourf_to_geojson(contourf=contourf)
        # Ensure geojson is a dictionary, not a string
        if isinstance(geojson, str):
            print("Converting GeoJSON string to dictionary...")
            import json
            geojson = json.loads(geojson)
    except Exception as e:
        print(f"Error creating GeoJSON: {e}")
        geojson = {"type": "FeatureCollection", "features": []}
    
    plt.close()  # Close figure to free memory
    
    return geojson, cmap.name  # Return both the GeoJSON and the colormap name

###########################
# CREATE INTERACTIVE MAP (Folium)
###########################
def create_folium_map(df_pred, df_real, geojson_data):
    """Create an interactive Folium map overlaying contours and real data points."""
    # Unpack the geojson and colormap name
    if isinstance(geojson_data, tuple) and len(geojson_data) == 2:
        geojson, cmap_name = geojson_data
    else:
        geojson = geojson_data
        cmap_name = "YlGnBu"  # Default colormap
    
    # Use centroid for the initial map view
    center_lat = df_real['Latitude'].mean()
    center_lon = df_real['Longitude'].mean()
    
    # Fallback to hardcoded values if needed
    if np.isnan(center_lat) or np.isnan(center_lon):
        center_lat, center_lon = 45, 10
    
    folium_map = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=4,  # Wider initial view
        tiles="cartodbpositron"
    )

    # Add contour overlay if available
    # Make sure geojson is a dictionary, not a string
    if isinstance(geojson, str):
        try:
            import json
            geojson = json.loads(geojson)
        except:
            print("Warning: Could not parse GeoJSON string")
            geojson = {"type": "FeatureCollection", "features": []}
    
    # Check if geojson is a dictionary with features
    if isinstance(geojson, dict) and "features" in geojson and geojson["features"]:
        folium.GeoJson(
            geojson, 
            style_function=lambda x: {
                'fillColor': x['properties']['fill'], 
                'opacity': 0.7,  # Slightly increased opacity
                'fillOpacity': 0.7,  # Increased opacity to make details more visible
                'weight': 0.5  # Thinner lines between contours for less visual interference
            }
        ).add_to(folium_map)
        
        # Add colorbar legend that matches the colormap used in the contours
        # Define color scales based on the colormap name
        color_scales = {
            "YlGnBu": ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'],
            "Blues": ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
            "BuPu": ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
            "Greens": ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
            "Reds": ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
        }
        
        # Use the appropriate color scale, defaulting to YlGnBu if not found
        colors = color_scales.get(cmap_name, color_scales["YlGnBu"])
        
        colormap = branca.colormap.LinearColormap(
            colors=colors, 
            index=np.linspace(0, 1, len(colors)),
            vmin=0,
            vmax=1,
            caption='Probability of Viviparity'
        )
        folium_map.add_child(colormap)
    else:
        print("Warning: No valid GeoJSON features found for contour overlay")
        
    # Create a separate layer for points showing exact probability values
    points_layer = folium.FeatureGroup(name="Sample Points (toggle on/off)")
    
    # Add a subset of points with probability values (to avoid cluttering the map)
    # Use systematic sampling to get a representative distribution
    sample_size = min(500, len(df_pred))  # Limit to 500 points max
    step = max(1, len(df_pred) // sample_size)
    
    df_sample = df_pred.iloc[::step].dropna(subset=["Probability"])
    
    for _, point in df_sample.iterrows():
        # Skip points without valid probability
        if np.isnan(point.get("Probability", np.nan)):
            continue
            
        # Create color based on probability
        prob_color = plt.cm.YlGnBu(point["Probability"])
        # Convert RGBA to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(prob_color[0]*255), 
            int(prob_color[1]*255), 
            int(prob_color[2]*255)
        )
        
        # Create popup with detailed information
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Grid Point</b><br>
            Lat: {point.Latitude:.4f}<br>
            Lon: {point.Longitude:.4f}<br>
            <b>Probability:</b> {point.Probability:.3f}<br>
            <hr style="margin: 5px 0;">
            <b>Environmental Data:</b><br>
        """
        
        # Add environmental variables
        for var in [col for col in point.index if col not in ['Latitude', 'Longitude', 'Probability', 'distance']]:
            if not pd.isna(point[var]):
                popup_html += f"{var}: {point[var]:.1f}<br>"
        
        popup_html += "</div>"
        
        # Add circle marker
        folium.CircleMarker(
            location=[point.Latitude, point.Longitude],
            radius=2,  # Small points
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(points_layer)
    
    # Add the points layer to the map but set it to off by default
    points_layer.add_to(folium_map)
    
    # Create separate layers for viviparous and oviparous species
    viviparous_layer = folium.FeatureGroup(name="Viviparous Species (1)", show=True)
    oviparous_layer = folium.FeatureGroup(name="Oviparous Species (0)", show=True)
    unknown_layer = folium.FeatureGroup(name="Unknown Reproductive Mode", show=True)
    
    # Add real data points, colored by their reproductive mode
    for _, row in df_real.iterrows():
        if np.isnan(row.Latitude) or np.isnan(row.Longitude):
            continue
            
        # Build popup content
        popup_text = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Observation Point</b><br>
            Lat: {row.Latitude:.4f}<br>
            Lon: {row.Longitude:.4f}<br>
        """
        
        if 'Species' in row:
            popup_text += f"<b>Species:</b> {row.Species}<br>"
        
        if 'IsViviparous' in row:
            if row.IsViviparous == 1:
                popup_text += "<b>Reproductive Mode:</b> Viviparous<br>"
            elif row.IsViviparous == 0:
                popup_text += "<b>Reproductive Mode:</b> Oviparous<br>"
            else:
                popup_text += "<b>Reproductive Mode:</b> Unknown<br>"
                
        popup_text += "</div>"
                
        # Create marker with appropriate color
        if 'IsViviparous' in row and row.IsViviparous == 1:
            # Viviparous - use darker blue
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.9
            ).add_to(viviparous_layer)
        elif 'IsViviparous' in row and row.IsViviparous == 0:
            # Oviparous - use lighter yellow/orange
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='orange',
                fill=True,
                fill_color='orange',
                fill_opacity=0.9
            ).add_to(oviparous_layer)
        else:
            # Unknown - use red or gray
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.9
            ).add_to(unknown_layer)
    
    # Add all layers to the map
    viviparous_layer.add_to(folium_map)
    oviparous_layer.add_to(folium_map)
    unknown_layer.add_to(folium_map)
    
    # Add a layer for predicted vs actual match assessment
    if 'IsViviparous' in df_real.columns and df_real['IsViviparous'].isin([0, 1]).any():
        assessment_layer = folium.FeatureGroup(name="Prediction Assessment", show=False)
        
        # For each observation point with known reproductive mode
        for _, row in df_real[df_real['IsViviparous'].isin([0, 1])].iterrows():
            # Find nearest prediction point
            df_pred['temp_dist'] = np.sqrt(
                (df_pred['Latitude'] - row.Latitude)**2 + 
                (df_pred['Longitude'] - row.Longitude)**2
            )
            nearest_pred = df_pred.loc[df_pred['temp_dist'].idxmin()]
            
            # Skip if no valid prediction
            if pd.isna(nearest_pred.get('Probability', np.nan)):
                continue
                
            # Calculate match percentage
            if row.IsViviparous == 1:
                match_pct = nearest_pred.Probability * 100
                correct_pred = nearest_pred.Probability >= 0.5
            else:  # IsViviparous == 0
                match_pct = (1 - nearest_pred.Probability) * 100
                correct_pred = nearest_pred.Probability < 0.5
            
            # Determine color based on match quality
            if correct_pred:
                # Good prediction - use green with intensity based on confidence
                color = f'#{int(155 + match_pct):02x}ff{int(155):02x}'
            else:
                # Poor prediction - use red with intensity based on error
                color = f'#ff{int(155 + (100-match_pct)):02x}{int(155):02x}'
            
            # Add marker showing prediction quality
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>Prediction Assessment</b><br>
                Actual: {"Viviparous" if row.IsViviparous == 1 else "Oviparous"}<br>
                Predicted Probability: {nearest_pred.Probability:.3f}<br>
                <b>Match: {match_pct:.1f}%</b><br>
                <b>Outcome: {"✓ Correct" if correct_pred else "✗ Incorrect"}</b>
            </div>
            """
            
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9
            ).add_to(assessment_layer)
        
        # Clean up temporary column
        if 'temp_dist' in df_pred.columns:
            df_pred.drop('temp_dist', axis=1, inplace=True)
            
        # Add the assessment layer to the map
        assessment_layer.add_to(folium_map)
        
        # Add a legend explaining the colors
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 90px; 
                    border:2px solid grey; z-index:9999; font-size:12px;
                    background-color: white; padding: 10px;
                    border-radius: 5px;">
            <span style="color: blue;"><b>●</b></span> Viviparous Species (1)<br>
            <span style="color: orange;"><b>●</b></span> Oviparous Species (0)<br>
            <span style="color: red;"><b>●</b></span> Unknown Reproductive Mode<br>
            <hr style="margin: 5px 0;">
            <i>Toggle layers using the control panel</i>
        </div>
        '''
        folium_map.get_root().html.add_child(folium.Element(legend_html))

    # Add map controls
    folium.LayerControl(collapsed=False).add_to(folium_map)
    plugins.Fullscreen().add_to(folium_map)
    plugins.MeasureControl().add_to(folium_map)
    
    # Save the map
    try:
        folium_map.save("viviparity_map_regres.html")
        print("✅ Map saved as 'viviparity_map.html'")
        
        # Also save a more detailed version
        folium_map.save("viviparity_map_detailed_regress.html")
        print("✅ Detailed map saved as 'viviparity_map_detailed.html'")
    except Exception as e:
        print(f"Error saving map: {e}")

###########################
# ANALYZE ENVIRONMENTAL DATA
###########################
def analyze_environmental_data(df_grid, df_real=None):
    """Perform additional analysis on environmental data."""
    # Get all environmental variables
    env_vars = [col for col in df_grid.columns if col not in ['Latitude', 'Longitude', 'Probability']]
    
    if len(env_vars) == 0:
        print("⚠️ No environmental variables found for analysis")
        return
    
    print("\n📊 Analyzing environmental data patterns...")
    
    # Create a grid for visualization
    try:
        # For each environmental variable, calculate statistics by latitude band
        lat_bands = pd.cut(df_grid['Latitude'], bins=10)
        lat_analysis = df_grid.groupby(lat_bands, observed=False)[env_vars].mean()
        lat_analysis.index = [f"{int(interval.left)}-{int(interval.right)}" for interval in lat_analysis.index]
        
        # Save latitude band analysis
        lat_analysis.to_csv("environmental_by_latitude.csv")
        print("✅ Latitude band analysis saved to environmental_by_latitude.csv")
        
        # For each environmental variable, calculate statistics by longitude band
        lon_bands = pd.cut(df_grid['Longitude'], bins=10)
        lon_analysis = df_grid.groupby(lon_bands, observed=False)[env_vars].mean()
        lon_analysis.index = [f"{int(interval.left)}-{int(interval.right)}" for interval in lon_analysis.index]
        
        # Save longitude band analysis
        lon_analysis.to_csv("environmental_by_longitude.csv")
        print("✅ Longitude band analysis saved to environmental_by_longitude.csv")
        
        # If we have real data points, compare environmental conditions at those points
        if df_real is not None and len(df_real) > 0:
            # For each real data point, extract nearest grid point's environmental data
            real_env_data = []
            
            for _, real_row in df_real.iterrows():
                # Calculate distance to each grid point
                df_grid['distance'] = np.sqrt(
                    (df_grid['Latitude'] - real_row['Latitude'])**2 + 
                    (df_grid['Longitude'] - real_row['Longitude'])**2
                )
                
                # Get closest point
                closest_idx = df_grid['distance'].idxmin()
                closest_point = df_grid.loc[closest_idx].copy()
                
                # Add real point info
                if 'Species' in real_row:
                    closest_point['Species'] = real_row['Species']
                
                # Add to collection
                real_env_data.append(closest_point)
            
            # Create DataFrame with environmental data at real points
            df_real_env = pd.DataFrame(real_env_data)
            
            # Save to CSV
            df_real_env.to_csv("environmental_at_real_points.csv", index=False)
            print("✅ Environmental data at real points saved to environmental_at_real_points.csv")
    
    except Exception as e:
        print(f"⚠️ Error in environmental analysis: {e}")

###########################
# MAIN
###########################
def main():
    try:
        print("Starting viviparity probability mapping...")
        
        # Load real data points
        df_real, min_lon, min_lat, max_lon, max_lat = load_real_data(CSV_FILE)
        
        # Load model
        model, scaler, features = load_model()
        
        # Create grid of points
        df_grid = make_lon_lat_grid(min_lon, min_lat, max_lon, max_lat, RESOLUTION)
        
        # Predict viviparity probability
        df_pred = predict_for_grid(df_grid, model, scaler, features)
        
        # Perform additional environmental analysis
        analyze_environmental_data(df_pred, df_real)
        
        # Create contours
        geojson_data = create_contours(df_pred)
        
        # Create interactive map
        create_folium_map(df_pred, df_real, geojson_data)
        
        print("\n✅ Process completed successfully!")
        print("📄 Output files:")
        print("  - environmental_values.csv (All environmental data by lat/long)")
        print("  - environmental_values_summary.csv (Summary statistics)")
        print("  - prediction_with_environmental_data.csv (Complete dataset with predictions)")
        print("  - environmental_by_latitude.csv (Environmental trends by latitude)")
        print("  - environmental_by_longitude.csv (Environmental trends by longitude)")
        print("  - environmental_at_real_points.csv (Environmental data at observation points)")
        print("  - viviparity_map.html (Interactive visualization)")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()