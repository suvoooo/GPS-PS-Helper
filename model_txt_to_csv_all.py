'''
modified from model_txt_to_csv.py
instead of one file pair, we create the csv for all
'''
import json
import pandas as pd
import re
from astropy.io import fits
from glob import glob
import os

# Directory containing the files
directory = './Example_Patches_JPR/'

# Find all txt and fits files
txt_files = glob(os.path.join(directory, '*_models.txt'))
fits_files = glob(os.path.join(directory, '*_counts.fits'))

# Function to extract number from filename
def extract_number(filename):
    match = re.search(r'\d+', filename)
    return match.group(0) if match else 'default'

# Pair files based on the extracted number
pairs = []
for txt_file in txt_files:
    number = extract_number(txt_file)
    corresponding_fits = next((f for f in fits_files if extract_number(f) == number), None)
    if corresponding_fits:
        pairs.append((txt_file, corresponding_fits))

# Process each pair
for txt_file, fits_file in pairs:
    # Get the patch center from the fits header
    fits_data = fits.open(fits_file)
    lon_c, lat_c = fits_data[0].header['CRVAL1'], fits_data[0].header['CRVAL2'] 
    fits_data.close()

    # Load model data from the txt file
    with open(txt_file, 'r') as file:
        data = json.load(file)

    # List to hold the data for the DataFrame
    model_data = []

    # Parse the data
    for model in data['properties']:
        model_name = model['name']
        spectral_index = [param['value'] for param in model['spectral']['parameters'] if param['name'] == 'index'][0]
        amplitude = [param['value'] for param in model['spectral']['parameters'] if param['name'] == 'amplitude'][0]
        E0 = [param['value'] for param in model['spectral']['parameters'] if param['name'] == 'reference'][0]
        lon = [param['value'] for param in model['spatial']['parameters'] if param['name'] == 'lon_0'][0]
        lat = [param['value'] for param in model['spatial']['parameters'] if param['name'] == 'lat_0'][0]
        if model['spatial']['type'] == 'PointSpatialModel': 
            sigma = 0.0
        else:
            sigma = [param['value'] for param in model['spatial']['parameters'] if param['name'] == 'sigma'][0]
        # sigma = [param['value'] for param in model['spatial']['parameters'] if param['name'] == 'sigma'][0]
        patch_center_lon, patch_center_lat = lon_c, lat_c
        model_data.append([model_name, spectral_index, amplitude, E0, lon, lat, sigma, patch_center_lon, patch_center_lat])

    # Create DataFrame
    df = pd.DataFrame(model_data, columns=['model-name', 'spectral-index', 'amplitude', 'E0', 
                                           'lon', 'lat', 'sigma', 'lon_c', 'lat_c'])

    # Save DataFrame to CSV using the number from the filename
    number = extract_number(txt_file)
    csv_filename = f'model_parameters_{number}.csv'
    df.to_csv(os.path.join(directory, csv_filename), index=False)

    print(f"CSV file '{csv_filename}' created successfully.")