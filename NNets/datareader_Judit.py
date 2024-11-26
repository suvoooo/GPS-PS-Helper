import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import random

from mpl_toolkits.axes_grid1 import make_axes_locatable

###########################
# we accumulate all the images and corresponding masks (npy files)
# leave 1/2 folders aside for performing the test
# we will use this part as a proxy for data reader 
# intend to add simple augmentation later 
###########################

# Loop through patches_v3 to patches_v7 for training and validation

def read_src_msk_npy(base_folder, train=True, sqrt_sc=False, log_sc=False):
    # Initialize lists for sources and masks (training/validation)
    source_files = []
    mask_files = []
    source_arrays = []
    mask_arrays = []
    if train:
        start_f, final_f = 3, 8
    else:
        start_f, final_f = 8, 9    
    for i in range(start_f, final_f):
        folder = os.path.join(base_folder, f'patches_v{i}')
        # Get all source and mask files in the folder
        sources = sorted(glob.glob(os.path.join(folder, 'patch_*_src_counts.npy')))
        masks = sorted(glob.glob(os.path.join(folder, 'patch_*_mask.npy')))
        source_files.extend(sources)
        mask_files.extend(masks)

        for src, msk in zip(sources, masks):
            src_data = np.load(src)
            msk_data = np.load(msk)
            # msk_data = 

            if sqrt_sc: src_data = np.sqrt(src_data)
            if log_sc:  src_data = np.log(src_data)    

            # apply the max norm
            src_data = src_data/(np.max(src_data)+1e-7) # pixel vals must be within 0-1    
            msk_data = msk_data/(np.max(msk_data)+1e-7)
            msk_data = 1 - msk_data

            source_arrays.append(src_data)
            mask_arrays.append(msk_data)


    return (source_files, mask_files, 
            np.array(source_arrays), np.array(mask_arrays))






# ##############################
# # Run this only once to check if
# # datareader is working or not
# ##############################


# # Define the base folder
# base_folder = '/d11/CAC/sbhattacharyya/Downloads/JPR_CTA_Check/patches/'



# (src_files, msk_files, src_arr, msk_arr) = read_src_msk_npy(base_folder)

# print ('how many files: ', len(src_files))

# # Check if sources and masks match in number
# assert len(src_files) == len(msk_files), "Mismatch in source and mask file counts."

# # Randomly select 3 source and corresponding mask files
# # check if selection and sorting by filenames work
# random_indices = random.sample(range(len(src_files)), 3)
# selected_sources = [src_files[i] for i in random_indices]
# selected_masks = [msk_files[i] for i in random_indices]

# # Plot the selected source and mask pairs
# fig, axs = plt.subplots(3, 3, figsize=(10, 9))

# for idx, (source_path, mask_path) in enumerate(zip(selected_sources, selected_masks)):
#     # Load the source and mask data
#     source_data = np.load(source_path)
#     mask_data = np.load(mask_path)
#     mask_data = mask_data/(np.max(mask_data))
#     mask_data = 1 - mask_data


    
#     # Plot source 1st bin on the left (column 0)
#     axs[idx, 0].imshow(np.sqrt(source_data[:, :, 0]), cmap='cool')
#     axs[idx, 0].set_title(f'Source {random_indices[idx]} [0.03 - 1 TeV]')
#     axs[idx, 0].axis('off')

#     axs[idx, 1].imshow(np.sqrt(source_data[:, :, 1]), cmap='cool')
#     axs[idx, 1].set_title(f'Source {random_indices[idx]} [1 - 10 TeV]')
#     axs[idx, 1].axis('off')
    
#     # Plot mask on the right (column 1)
#     im3 = axs[idx, 2].imshow(mask_data[:, :,], cmap='gray')
#     axs[idx, 2].set_title(f'Mask {random_indices[idx]}')
#     axs[idx, 2].axis('off')
#     # Add colorbar for the mask
#     divider3 = make_axes_locatable(axs[idx, 2])
#     cax3 = divider3.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(im3, cax=cax3)

# # Display the plot
# plt.tight_layout()
# plt.savefig('./random_patch_mask.png', dpi=200)
# # plt.show()

