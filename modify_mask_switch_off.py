'''
code that takes in the mask npy file and corresponding csv file 
containing source info including the pixel coordinates. 

Switches off randomly certain percentages of mask
'''

import pandas as pd
import numpy as np
import os, glob, re
from astropy.io import fits

file_path = './Example_Patches_JPR/'

all_csvs = [f for f in os.listdir(file_path) if f.endswith('w_xy.csv')]
print (all_csvs)

mk_path = file_path



for c_f in all_csvs:
    print ('started processing data for: ', c_f)

    # Extract number from the CSV filename
    match = re.search(r'(\d+)', c_f)
    if match:
        p_number = match.group(1) # returns the patch number
    else:
        print(f"Could not extract number from filename {c_f}")
        continue

    print ('check the corresponding patch number extracted: ', p_number)

    patch_df = pd.read_csv(os.path.join(file_path, c_f))

    patch_df['on_off'] = 1 # default status : Mask On
    
    num_sources = patch_df.shape[0]
    print ('total number of sources: ', num_sources)
    percent = 0.3 # how many percentages will be switched off
    percent_s = '0d30'

    n_modify = max(1, int(percent * num_sources))
    print ('how many masks will be switched off: ', n_modify)

    # Randomly select objects
    modify_indices = np.random.choice(patch_df.index, n_modify, replace=False)
    print ('what is modify indices:', modify_indices)

    patch_df.loc[modify_indices, 'on_off'] = 0 # mask switched off

    mask_filename = f'patch_{p_number}_masks.fits'
    mask_filename_mod = f'patch_{p_number}_masks_{percent_s}Off.npy'

    mask_path = os.path.join(mk_path, mask_filename)
    mod_mask_path = os.path.join(mk_path, mask_filename_mod)

    if os.path.exists(mask_path): # if the file exists do operation
        mk_data = fits.open(mask_path)
        mk_data = mk_data[0].data
        mk_data = np.moveaxis(mk_data, 0, -1) # reorder axis 

        for idx in modify_indices:
            obj = patch_df.loc[idx]

            x_center, y_center = obj['x'], obj['y']
            sigma = obj['sigma'] *100 
            # multiply with hundred because spatial binsize is 0.01
            # so sigma = 0.11 ~ 11 pixels (gives 1 sigma estimate)
            sigma_3 = int(sigma * 1.732) #(multiply with sqrt 3 to have 3 sigma estimate)
            # calculate boundaries
            x_start = max(0, int(x_center) - sigma_3)
            x_end = min(mk_data.shape[1], int(x_center) + sigma_3)

            y_start = max(0, int(y_center) - sigma_3)
            y_end = min(mk_data.shape[0], int(y_center)  + sigma_3)

            ### set the mask to max value
            # for the current dataset
            # min value signifies pixels belonging to src, max for bkg

            mk_data[y_start:y_end, x_start:x_end] = mk_data.max()

        np.save(mod_mask_path, mk_data)
    else: 
        print (f"mask file not found: {mask_path}")
    
    patch_df.to_csv(file_path + f'patch-{p_number}-sources-w_xy_{percent_s}Off.csv', index=False)




