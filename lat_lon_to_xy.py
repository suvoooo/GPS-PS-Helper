'''

'''

import pandas as pd
import numpy as np

from gammapy.maps import MapAxis, WcsGeom


start_index = 0
end_index = 1

path_to_file = './Example_Patches_JPR/'

for index in range(start_index, end_index + 1):
    df = pd.read_csv(path_to_file + f'model_parameters_{index}.csv')
    lon_coords = df['lon'].to_numpy()
    lat_coords = df['lat'].to_numpy()
    # pixel_coords = np.load(f'coords_log_{index}.npy')
    print(f"Processing index {index}")
    print (lat_coords.shape, lon_coords.shape)

    # x_coords = pixel_coords[:, 0]
    # y_coords = pixel_coords[:, 1]
    lon_c, lat_c = df['lon_c'].iloc[0], df['lat_c'].iloc[0]

    geom = WcsGeom.create(npix=(512, 512), skydir=(lon_c, lat_c), 
                          binsz=0.01,  
                          frame="galactic", )
    
    lat_lon_tuple = (lon_coords, lat_coords)
    x, y = geom.coord_to_pix(lat_lon_tuple)
    print (type(x), x.shape, x[0:5], y[0:5])

    # x = x.value
    # y = y.value

    df['x'] = x
    df['y'] = y

    df.to_csv(path_to_file + f'patch_{index}-sources-w_xy.csv', index=False)

    


