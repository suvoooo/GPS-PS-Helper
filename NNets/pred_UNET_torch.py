import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


import neural_nets_torch, datareader_Judit


folder_path = '/d11/CAC/sbhattacharyya/Downloads/JPR_CTA_Check/patches/all_patches/'


num_samples = 844 # get it from the training script

################################
# test data preparation
################################


(im_list, mk_list, 
 im_arr, mk_arr) = datareader_Judit.read_src_msk_npy(folder_path, 
                                                     train=False, sqrt_sc=True, log_sc=False)
# im_list : list of filenames
# im_arr  : list of corresponding image arrays

print ('check len of src and msks: ', len(im_list), 
       len(mk_arr))

print ('check shape of an img arr: ', im_arr[0].shape)




###########################
# convert to tensors from numpy arrays
###########################

epochs=200
batch_size=16 # very low batch_size due to low gpu

X_test_tensors = torch.tensor(im_arr, dtype=torch.float32)
y_test_tensors = torch.tensor(mk_arr, dtype=torch.float32)

# X_val_tensor = torch.tensor(test_ims, dtype=torch.float32)
# y_val_tensor = torch.tensor(test_msk, dtype=torch.float32)


print("X_train tensor shape:", X_test_tensors.shape)
# shape here (N, H, W, C), good for tensorflow but not for torch
print("y_train tensor shape:", y_test_tensors.shape)
# print("X_val tensor shape:", X_val_tensor.shape)
# print("y_val tensor shape:", y_val_tensor.shape)




#########################
# reshape to have the channel first:
#########################

X_test_tensors = X_test_tensors.permute(0, 3, 1, 2).float()
y_test_tensors = y_test_tensors.permute(0, 3, 1, 2).float()

# X_val_tensor   = X_val_tensor.permute(0, 3, 1, 2).float()
# y_val_tensor   = y_val_tensor.permute(0, 3, 1, 2).float()

print("X_train tensor shape after permute:", X_test_tensors.shape)
# shape here (N, C, H, W), good for torch
print("y_train tensor shape after permute:", y_test_tensors.shape)


# Check for NaNs in PyTorch tensors
print("NaNs in PyTorch training images:", torch.isnan(X_test_tensors).any())
print("NaNs in PyTorch training masks:", torch.isnan(y_test_tensors).any())
# print("NaNs in PyTorch validation images:", torch.isnan(X_val_tensor).any())
# print("NaNs in PyTorch validation masks:", torch.isnan(y_val_tensor).any())



###################################
## define the loss
## for now go with combo
## switch between bce and combo
###################################

#######################
# Combo loss
#######################
def dice_loss(y_true, y_pred, smooth=1e-4):
 intersection = torch.sum(y_true * y_pred, dim=(0, 1, 2))
 union = torch.sum(y_true, dim=(0, 1, 2)) + torch.sum(y_pred, dim=(0, 1, 2))
 dice = (2. * intersection + smooth)/(union + smooth)
 return 1-dice.mean() # from the stack post (https://stackoverflow.com/questions/72195156/correct-implementation-of-d>

bce_loss_torch = torch.nn.BCEWithLogitsLoss()
# sadly bceloss gives nan outputs (need to inspect)#30122024

def bce_loss(y_true, y_pred):
 #y_pred = torch.clamp(y_pred, min=1e-6, max=1-1e-6)
 return bce_loss_torch(y_true, y_pred)
 # return F.binary_cross_entropy(y_true, y_pred)


def combo_loss(y_true, y_pred, alpha=0.6, smooth=1e-4):
 # y_pred = torch.clamp(y_pred, min=1e-6, max=1-1e-6)
 dice = dice_loss(y_true, y_pred, smooth)
 bce  = bce_loss(y_true, y_pred)
 combined = alpha * bce + (1-alpha) * dice
 return combined




########################
# dice coeff
########################
def dice_coeff(y_true, y_pred, smooth=1e-4): 
 intersection = torch.sum(y_true * y_pred, dim=(0, 1, 2))
 union = torch.sum(y_true, dim=(0, 1, 2)) + torch.sum(y_pred, dim=(0, 1, 2))
 dice = (2. * intersection + smooth)/(union + smooth)
 return 1-dice.mean()



###################################
# Load the UNET model
# we use the helper neural_nets
###################################
loaded_model = neural_nets_torch.AllModels(h=512, w=512, bins=3, 
                                           activation_f='sigmoid', 
                                           activation_i='gelu', dropout_p=0.2)

dummy_input = torch.randn(1, 3, 512, 512)
print(summary(loaded_model, input_data=dummy_input, verbose=1))

##########################
# Optim & Loss
##########################
learning_rate = 3e-5

# optimizer = torch.optim.Adam(loaded_model.parameters(), lr=learning_rate)
# criterion = combo_loss
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, 
#                               min_lr=2e-8, patience=5, )

model_save_path = folder_path + 'torch_unet_JPR_%dSamples_check_SqrtSc.pth'%(num_samples)

# model_save_path = folder_path + 'torch_unet_bright_check.pth'

loaded_model.load_state_dict(torch.load(model_save_path))

loaded_model.eval() # set for evaluation mode

print ('!!!!! model loaded for evaluation mode !!!!!')




##########################
# a visualizer for the predictions
##########################

# Function to plot images, true masks, and predicted masks side by side
def plot_prediction_comparison(image, true_mask, predicted_mask, idx, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(12, 5))

    # Plot original image
    plt.subplot(1, 4, 1)
    im1 = axs[0].imshow(image[:, :, 0], cmap='inferno')
    plt.title(f'Image {idx}: Bin0')
    plt.axis('off')
    # Add colorbar for the original image
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax0)

    # Plot original image
    plt.subplot(1, 4, 2)
    im2 = axs[1].imshow(image[:, :, 2], cmap='inferno')
    plt.title(f'Image {idx}: Bin2')
    plt.axis('off')
    # Add colorbar for the original image
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1)

    # Plot true mask
    plt.subplot(1, 4, 3)
    im3 = axs[2].imshow(true_mask[:, :], cmap='gray')
    plt.title(f'True Mask {idx}')
    plt.axis('off')
    # Add colorbar for the original mask
    divider2 = make_axes_locatable(axs[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax2)

    # Plot predicted mask
    plt.subplot(1, 4, 4)
    im4 = axs[3].imshow(predicted_mask[:, :], cmap='gray')
    plt.title(f'Predicted Mask {idx}')
    plt.axis('off')
    # Add colorbar for the pred mask
    divider3 = make_axes_locatable(axs[3])
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im4, cax=cax3)

    plt.tight_layout()
    plt.savefig(f'{save_path}/Image-Mask-Pred{idx}_{num_samples}Samp_th0d4_SqrtSc.png', dpi=200)
    print ('figure saved: ', idx)
    plt.close()
    # plt.show()




##########################################
# make predictions and visualize images
#########################################

# print ('check filename: ', im_list[0])
save_pred_path = folder_path + 'results_torch'
random_indices = random.sample(range(len(im_list)), 3) 
print ('chosen number of ims: ', random_indices)
# we select only 3 images

for idx in random_indices:
  # try to get the current patch number
  num = int(im_list[idx].split('/')[-1].split('_')[1])
  # Predict for the current stacked image 
  input_tensor = X_test_tensors[idx].unsqueeze(0)
  inp_im  = im_arr[idx]
  tr_mk   = mk_arr[idx]
  print ('input tensor just before preds: ', input_tensor.shape)
  # perform prediction: 
  with torch.no_grad():
       prediction_tensor = loaded_model(input_tensor)
       predictions = prediction_tensor.cpu().numpy() 
  grid2D_pred = predictions[0, 0, :, :] # change here from tf (first batch, second ch) 
  print('grid2D_pred.shape: ', grid2D_pred.shape) 
  print('grid2D avg: ', np.mean(grid2D_pred))

  grid2D_pred_th = (grid2D_pred > 0.7).astype(np.uint8) # tends to predict brighter pixels?

  plot_prediction_comparison(inp_im, tr_mk, predicted_mask=grid2D_pred_th, idx=num, save_path=save_pred_path)




