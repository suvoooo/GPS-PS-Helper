'''
a data reader module that 
access the background and corresponding source files 
given the file index, we will return the source only image, bkg only image and full image

the fits file have 3 energy bins for all cases, so our final images would be of shape (W, H, 3)
'''

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


from astropy.io import fits


def fits_to_arr(fits_file, channel=3, sqrt=1):
    fits_data = fits.open(fits_file)
    im_data = fits_data[0].data.astype(np.float32)
    # print (im_data.shape)
    im_data = np.moveaxis(im_data, 0, -1)
    # print (im_data.shape) # shape here is (512, 512, 3)
    if channel==1:
        im_data = im_data = im_data[:, :, 1:2]
    elif channel==0:
        im_data = im_data[:, :, 0:1]    
    else:
        im_data = im_data[:, :, 2:3]    
    if sqrt==1:
        im_data = np.sqrt(im_data)

    return im_data


def read_fits_data(base_dir:str, channel:int, sqrt:int, train:bool):
    source_dict, source_files = {}, {}
    bkg_dict, bkg_files = {}, {}
    total_dict, total_files = {}, {}
    # nums_set = set()

    ## define the number range train and test sets
    if train: 
        num_range = set(range(1, 557)) # according to zoja's set
    else:
        num_range = set(range(557, 576)) # test set


    # Loop through each subdirectory and process files
    for subdir in ['Total_Asi_JPR', 'bkg_model', 'x_data_new']:
        folder_path = os.path.join(base_dir, subdir)
        for filename in os.listdir(folder_path):
            if filename.endswith(".fits"):
                # Extract the number from the filename
                number = int(filename.split('-')[1].split('.')[0])
                if number not in num_range:
                    continue
                # Read the FITS file
                file_path = os.path.join(folder_path, filename)
                # print ('check the file path: ', file_path)
                # with fits.open(file_path) as hdul:
                #     image_data = hdul[0].data
                image_data = fits_to_arr(file_path, channel=channel, sqrt=sqrt)
                # print ('shape here should be (512, 512, 3): ', image_data.shape)
                
                # Append data to appropriate list
                if subdir == 'Total_Asi_JPR':
                    source_dict[number] = image_data
                    source_files[number] = filename
                elif subdir == 'bkg_model':
                    bkg_dict[number] = image_data
                    bkg_files[number] = filename
                elif subdir == 'x_data_new':
                    total_dict[number] = image_data
                    total_files[number] = filename
                # nums_set.add(number)    
    
    valid_nums = set(source_dict.keys()) & set(bkg_dict.keys()) & set(total_dict.keys())
    print ('len valid nums: ', len(valid_nums))

    nums_list = sorted(valid_nums)
    # Sort all lists by nums_list to ensure corresponding indices match across lists
    # This sorting ensures that source, background, and total images correspond correctly
    # source_list = [x for _, x in sorted(zip(nums_list, source_list))]
    # source_files = [x for _, x in sorted(zip(nums_list, source_files))]

    source_list = [source_dict[num] for num in nums_list ]
    source_files_list = [source_files[num] for num in nums_list]


    # bkg_list = [x for _, x in sorted(zip(nums_list, bkg_list))]
    # bkg_files = [x for _, x in sorted(zip(nums_list, bkg_files))]

    bkg_list = [bkg_dict[num] for num in nums_list ]
    bkg_files_list = [bkg_files[num] for num in nums_list ]

    # total_list = [x for _, x in sorted(zip(nums_list, total_list))]
    # total_files = [x for _, x in sorted(zip(nums_list, total_files))]

    total_list = [total_dict[num] for num in nums_list ]
    total_files_list = [total_files[num] for num in nums_list ]

    # nums_list.sort()

    return (source_list, bkg_list, total_list, nums_list, source_files_list, 
            bkg_files_list, total_files_list)    


def read_npy_data(base_dir:str, channel:int, sqrt:int, train:bool):
    source_dict, source_files = {}, {}
    total_dict, total_files = {}, {}
    # nums_set = set()

    if train: 
        num_range = set(range(1, 557))
    else:
        num_range = set(range(557, 576))    


    for subdir in ['only_sources', 'x_data_new_npy']:
        folder_path = os.path.join(base_dir, subdir)
        for filename in os.listdir(folder_path):
            if filename.endswith(".npy"):
                # Extract the number from the filename
                number = int(filename.split('_')[2].split('.')[0])
                if number not in num_range:
                    continue
                file_path = os.path.join(folder_path, filename)
                im_data = np.load(file_path)
                if channel==0 or channel==1:
                    im_data = im_data[:, :, channel:channel+1]
                else: im_data = im_data[:, :, channel]

                if sqrt==1:
                    im_data = np.sqrt(im_data)
                else:
                    im_data = im_data

                if subdir=='only_sources':
                    source_dict[number] = im_data
                    source_files[number] = filename
                elif subdir=='x_data_new_npy':
                    total_dict[number] = im_data
                    total_files[number] = filename
                
    valid_nums = set(source_dict.keys()) &  set(total_dict.keys())
    print ('len valid nums: ', len(valid_nums))
    nums_list = sorted(valid_nums)

    source_list = [source_dict[num] for num in nums_list if num in source_dict]
    source_files_list = [source_files[num] for num in nums_list if num in source_files]


    # total_list = [x for _, x in sorted(zip(nums_list, total_list))]
    # total_files = [x for _, x in sorted(zip(nums_list, total_files))]

    total_list = [total_dict[num] for num in nums_list if num in total_dict]
    total_files_list = [total_files[num] for num in nums_list if num in total_files]

    # nums_list.sort()

    return (source_list, total_list, nums_list, source_files_list, total_files_list)                        



def read_npy_data_ZR_JP(base_dir:str, channel:int, sqrt:int):
    '''
    use Zoja's poisson data and Judit's assimov sources
    '''
    source_dict, source_files = {}, {}
    total_dict, total_files = {}, {}
    nums_set = set()
    for subdir in ['only_sources', 'x_data_new_ZR_Poi_npy']:
        folder_path = os.path.join(base_dir, subdir)
        print ('check current folder: ', folder_path)
        for filename in os.listdir(folder_path):
            if filename.endswith(".npy"):
                # Extract the number from the filename
                number = int(filename.split('_')[2].split('.')[0])
                if number==0 or number==599:
                    continue
                file_path = os.path.join(folder_path, filename)
                im_data = np.load(file_path)
                if channel==0 or channel==1:
                    im_data = im_data[:, :, channel:channel+1]
                else: im_data = im_data[:, :, channel]

                if sqrt==1:
                    im_data = np.sqrt(im_data)
                else:
                    im_data = im_data

                if subdir=='only_sources':
                    source_dict[number] = im_data
                    source_files[number] = filename
                elif subdir=='x_data_new_ZR_Poi_npy':
                    total_dict[number] = im_data
                    total_files[number] = filename
                nums_set.add(number)

    nums_list = sorted(nums_set)

    source_list = [source_dict[num] for num in nums_list if num in source_dict]
    source_files_list = [source_files[num] for num in nums_list if num in source_files]


    # total_list = [x for _, x in sorted(zip(nums_list, total_list))]
    # total_files = [x for _, x in sorted(zip(nums_list, total_files))]

    total_list = [total_dict[num] for num in nums_list if num in total_dict]
    total_files_list = [total_files[num] for num in nums_list if num in total_files]

    # nums_list.sort()

    return (source_list, total_list, nums_list, source_files_list, total_files_list)





### run the code # just for test

path_to_dir='/d11/CAC/sbhattacharyya/Downloads/Zoja_CTA_Check/new_data/Zoja-Asi-JPR/'

# src_list, bkg_list, tot_list, num_list, src_files, bkg_files, tot_files = read_fits_data(path_to_dir, channel=1, sqrt=1)


# print ('check lens of the lists@ ', len(src_list), len(bkg_list), len(tot_list), len(num_list))
# print ('check shape of a file src, bkg, tot, num: ', src_list[5].shape, bkg_list[5].shape, 
#        tot_list[5].shape, )

# print ('check corresponding file number: ', num_list[5])
# print ('check corresponding source file: ', src_files[5])
# print ('check corresponding bkg file: ', bkg_files[5])
# print ('check corresponding tot file: ', tot_files[5])

# src_list, tot_list, num_list, src_files, tot_files = read_npy_data(path_to_dir, channel=1, sqrt=1)

# print ('check lens of the lists@ ', len(src_list), len(tot_list), len(num_list))
# print ('check shape of a file src, bkg, tot, num: ', src_list[5].shape, tot_list[5].shape, )

# print ('check corresponding file number: ', num_list[5])
# print ('check corresponding source file: ', src_files[5])
# print ('check corresponding tot file: ', tot_files[5])

# src_list, tot_list, num_list, src_files, tot_files = read_npy_data_ZR_JP(path_to_dir, channel=1, sqrt=1)

# print ('check lens of the lists@ ', len(src_list), len(tot_list), len(num_list))
# print ('check shape of a file src, bkg, tot, num: ', src_list[5].shape, tot_list[5].shape, )

# print ('check corresponding file number: ', num_list[5])
# print ('check corresponding source file: ', src_files[5])
# print ('check corresponding tot file: ', tot_files[5])