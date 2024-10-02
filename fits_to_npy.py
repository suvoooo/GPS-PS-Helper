'''
read and save the fits files in npy format
easier to handle, smaller & faster


'''

from astropy.io import fits
import numpy as np
import os, glob
import re


class FitstoNumpySave:
    def __init__(self, directory):
        self.directory = directory
        self.fits_files_bkg = glob.glob(os.path.join(directory, '*_bkg.fits'))
        self.fits_files_src = glob.glob(os.path.join(directory, '*_counts.fits'))
        self.fits_files_msk = glob.glob(os.path.join(directory, '*_masks.fits'))

    @staticmethod
    def extract_number(filename): # Function to extract number from filename
        match = re.search(r'\d+', filename)
        return match.group(0) if match else 'default'
    
    @staticmethod
    def fits_to_arr(fits_file):
        '''
        function that open gammapy fits map 
        and returns an array
        '''
        fits_data = fits.open(fits_file)
        im_data = fits_data[0].data.astype(np.float32)
        # print (im_data.shape)
        im_data = np.moveaxis(im_data, 0, -1)
        # print (im_data.shape) # shape here is (512, 512, 3) # (512, 512, 1)
        return im_data

    def save_fits_as_npy(self, fits_type='bkg'):
        '''
        function that takes the fits paths as input
        and save the fits data in a numpy array
        '''

        if fits_type == 'bkg':
            fits_files_paths = self.fits_files_bkg
            comp=0
        elif fits_type == 'src':
            fits_files_paths = self.fits_files_src
            comp=1
        else:
            fits_files_paths = self.fits_files_msk
            comp=2


        for fits_file in fits_files_paths:

            im_data = self.fits_to_arr(fits_file)
            match_n = self.extract_number(fits_file)
            if comp==0: # bkg
                save_filename = f'patch_{match_n}_bkg_counts.npy'
            elif comp==1: #src
                save_filename = f'patch_{match_n}_src_counts.npy'
            else:
                save_filename = f'patch_{match_n}_mask.npy'    

            np.save(os.path.join(self.directory, save_filename), im_data)
            print (f"check_saved file: {save_filename}")

processor = FitstoNumpySave('./Example_Patches_JPR/')
processor.save_fits_as_npy(fits_type='bkg') # bkg fits
processor.save_fits_as_npy(fits_type='src') # src fits
processor.save_fits_as_npy(fits_type='msk') # mask fits


