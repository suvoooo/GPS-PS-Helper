import neural_nets
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

import tensorflow as tf
from skimage.transform import resize
from astropy.io import fits
# from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import datareader_Judit

folder_path = '/d11/CAC/sbhattacharyya/Downloads/JPR_CTA_Check/patches/all_patches/'

###############################
# load src and corresponding masks
################################

(im_list, mk_list, 
 im_arr, mk_arr) = datareader_Judit.read_src_msk_npy(folder_path, 
                                                     train=False, sqrt_sc=True, log_sc=False)
# im_list : list of filenames
# im_arr  : list of corresponding image arrays

print ('check len of src and msks: ', len(im_list), 
       len(mk_arr))

###################################
# dice coeff
###################################
def dice_coeff(y_true, y_pred, smooth=1e-4):
    intersection = K.sum(y_true * y_pred, axis=(0, 1))
    union = K.sum(y_true, axis=(0, 1)) + K.sum(y_pred, axis=(0, 1))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1-dice
    
    
#######################
# Combo loss
#######################
def dice_loss(y_true, y_pred, smooth=1e-4):
 intersection = K.sum(y_true * y_pred, axis=(0, 1))
 union = K.sum(y_true, axis=(0, 1)) + K.sum(y_pred, axis=(0, 1))
 dice = (2. * intersection + smooth)/(union + smooth)
 return 1-dice # from the stack post (https://stackoverflow.com/questions/72195156/correct-implementation-of-d>

def bce_loss(y_true, y_pred):
 return tf.keras.losses.binary_crossentropy(y_true, y_pred)


def combo_loss(y_true, y_pred, alpha=0.5, smooth=1e-4):
 dice = dice_loss(y_true, y_pred, smooth)
 bce  = bce_loss(y_true, y_pred)
 combined = alpha*bce + (1-alpha)*dice
 return combined


loaded_model = neural_nets.all_models(h=512, w=512, bins=3, activation_f='sigmoid')
unet = loaded_model.unet_model(act='gelu')
print ('check model summary: ', '\n')
unet.summary()


unet.load_weights(folder_path + 'CTA_JPR_Patches_844samples_200ep_ComboLoss_actGelu_SqrtSc.weights.h5', )
                               #custom_objects={'combo_loss': combo_loss, 'dice_coeff':dice_coeff})


print ('model ready for preds')

##########################
# a visualizer for the predictions
##########################

# Function to plot images, true masks, and predicted masks side by side
def plot_prediction_comparison(image, true_mask, predicted_mask, idx, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(12, 4))

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
    plt.savefig(f'{save_path}/results_tf/Image-Mask-Pred{idx}_844Samp_th0d4_SqrtSc.png', dpi=200, bbox_inches='tight')
    plt.close()
    # plt.show()

#############
# plot predictions for random selected images
#############

random_indices = random.sample(range(len(im_arr)), 3)

for idx in random_indices:
  image = im_arr[idx]
  true_mask = mk_arr[idx]
  predictions = unet.predict(np.expand_dims(image, axis=0))
  grid2D_pred = predictions[0, :, :, 0]
  grid2D_pred_th = (grid2D_pred > 0.5).astype(np.uint8)
  print ('grid2D shape: ', grid2D_pred.shape)
  print ('grid 2D avg: ', np.mean(grid2D_pred))
  plot_prediction_comparison(image, true_mask, grid2D_pred_th, idx, folder_path)

