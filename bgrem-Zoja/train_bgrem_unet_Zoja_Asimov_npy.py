'''
vresion tf 2.17.0

instead of diffusion, we use a simple denoising autoencoder

denoising: starting from Asimov to reach sources only image
removes CR background; 

fits converted to npy files

checked for both channel 0, 1; 

customised ssim loss
'''

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import datareader_bkg_src_Asi

from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

# tf.compat.v1.enable_eager_execution()





# data
num_epochs = 100 
image_size = 512
cutout_size = [image_size,image_size]


# optimization
batch_size = 32
start_learning_rate = 1e-4
learning_rate_decay = 0.90
weight_decay = 1e-4



###############################
# load the source, bkg and tot lists
###############################

path_to_dir='/d11/CAC/sbhattacharyya/Downloads/Zoja_CTA_Check/new_data/Zoja-Asi-JPR/'

(src_list, tot_list, num_list, 
 src_files, tot_files) = datareader_bkg_src_Asi.read_npy_data(path_to_dir, 
                                                              channel=0, sqrt=1, train=True)

# src here is the asimov image (total)
# tot here is the  poisson image (total)


print ('check the number of ims in different lists: ', len(src_list), len(tot_list))
print ('check shape: ', src_list[0].shape) # shape here is (512, 512, 1)

print ('check example src counts: ', src_list[0].sum())
print ('check example tot counts: ', tot_list[0].sum())

print ('check consistency in filenames: src:', src_files[3], src_files[300], src_files[497])
print ('check consistency in filenames: total:', tot_files[3], tot_files[300], tot_files[497])



####################################################
# normalize the images by dividing with max count
####################################################
def preprocess_max_norm(image): # , target_size=(512, 512)
    # Convert to a TensorFlow tensor if it's not already
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Normalize the image by dividing by the maximum pixel value
    max_pixel_value = tf.reduce_max(image)
    max_pixel_value = tf.maximum(max_pixel_value, 1e-9)
    # gaussian scaling
    
    # Normalize # 
    normalized_image = image / max_pixel_value
    
    return normalized_image, max_pixel_value


#####################
# if we add augmentation
#####################

def rotate_lr(image):
    return tf.image.random_flip_left_right(image)

def rotate_ud(image):
    return tf.image.random_flip_up_down(image)


# need to apply the max norm after augmentation

def preprocess_and_augment(image): # apply this after max_norm is applied to the dataset
    norm_image, max_pix_val = preprocess_max_norm(image)
    image_lr = rotate_lr(image)
    image_ud = rotate_ud(image_lr)
    return image_ud, max_pix_val



################################
# Creating tfds for faster reading
################################

def get_datasets(source_list, total_list, batch_size):
    # Convert lists to TensorFlow datasets
    source_ds = tf.data.Dataset.from_tensor_slices(source_list)
    # bkg_ds = tf.data.Dataset.from_tensor_slices(bkg_list)
    total_ds = tf.data.Dataset.from_tensor_slices(total_list)

    # Map the normalization and resize function
    source_ds = source_ds.map(lambda img: preprocess_max_norm(img), num_parallel_calls=tf.data.AUTOTUNE)
    total_ds = total_ds.map(lambda img: preprocess_max_norm(img), num_parallel_calls=tf.data.AUTOTUNE)

    # Separate the image and max_pixel_value into two datasets
    source_images_ds = source_ds.map(lambda img, max_val: img)
    source_max_vals_ds = source_ds.map(lambda img, max_val: max_val)
    total_images_ds = total_ds.map(lambda img, max_val: img)
    total_max_vals_ds = total_ds.map(lambda img, max_val: max_val)

    
    # Assuming you want to use the 'total' images as inputs and the 'source' images as targets
    input_ds = total_images_ds
    target_ds = source_images_ds

    # Combine the input and target datasets
    dataset = tf.data.Dataset.zip(((input_ds, target_ds), (total_max_vals_ds, source_max_vals_ds)))

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(source_list))

    # Splitting the dataset into training, validation, and testing
    # Assuming 70% training, 15% validation, 15% test
    train_size = int(0.82 * len(source_list))
    val_size = int(0.13 * len(source_list))

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    # Batch and prefetch the datasets
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset





def create_image_only_dataset(dataset):
    """
    Strips out the max values from the dataset, keeping only the image pairs for model training.
    """
    image_only_dataset = dataset.map(lambda images, max_vals: images)
    return image_only_dataset




# Example usage
train_ds, val_ds, test_ds = get_datasets(src_list, tot_list, batch_size)


#########################################

# # To check the shape of the images in your dataset
# for (images, labels), _ in train_ds.take(1):  # Just taking one batch from the dataset
#     print('Input image shape:', images.shape)
#     print('Label image shape:', labels.shape)

train_ds_np = tfds.as_numpy(train_ds)

print ('check dataset type: ', type(train_ds))
print ('check shape: ', train_ds_np, '\n', )

# Use the function to create new datasets for training and validation
train_image_ds = create_image_only_dataset(train_ds)
val_image_ds = create_image_only_dataset(val_ds)

print ('check val image ds: ', val_image_ds)

#########################
# Let's randomly plot some total and source images
#########################

def plot_sample_images(train_dataset, num_samples=3):
    # Make an iterator from the training dataset
    iterator = iter(train_dataset)
    
    # Retrieve the first batch (assuming batch size is at least as large as num_samples)
    ((total_images, source_images), (tot_max_vals, src_max_vals)) = next(iterator)
    
    # Randomly select indices for the images to display
    indices = np.random.choice(range(total_images.shape[0]), size=num_samples, replace=False)
    
    # Set up the plot - 2 rows and 3 columns
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 10))
    
    for i, idx in enumerate(indices):
        # Plot total image
        ax1 = axes[0, i]
        im1 = ax1.imshow(np.sqrt(total_images[idx].numpy()[:, :, 0]), cmap='inferno')  # Adjust color map or normalization as needed
        ax1.axis('off')  # Hide axes
        ax1.set_title(f'Total Image: {idx+1}; Bin1')
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label(r'$\sqrt{N}$', fontsize=10)
        # Plot source image
        ax2 = axes[1, i]
        im2 = ax2.imshow(np.sqrt(source_images[idx].numpy()[:, :, 0]), cmap='inferno')  # Adjust color map or normalization as needed
        ax2.axis('off')  # Hide axes
        ax2.set_title(f'Source Image: {idx+1}; Bin1')
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label(r'$\sqrt{N}$', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./random_image_src-OnlyAsi_tot_Ch0.png', dpi=200)

    # plt.show()

plot_sample_images(train_ds)


########################################
# if we add a resnet block
########################################
def resnet_block(inp_tensor, num_filters):
    x1 = keras.layers.Conv2D(num_filters, (3, 3), padding='same', 
                            activation='gelu')(inp_tensor)
    x1 = keras.layers.Conv2D(num_filters, (3, 3), padding='same', 
                            activation='gelu')(x1)
    
    x = keras.layers.Conv2D(num_filters, (3, 3), padding='same', 
                            activation='gelu')(inp_tensor)
    
    out = keras.layers.Add()([x, x1])
    out = keras.layers.Activation('gelu')(out)

    return out



###################################
# aec (unet-like) for denoising (map to map)
###################################

def denoising_unet(input_shape=(512, 512, 1)):
    inputs = keras.layers.Input(shape=input_shape)

    # encoding block
    c1 = keras.layers.Conv2D(32, (3, 3), activation='gelu', padding='same')(inputs)
    c1 = keras.layers.Conv2D(32, (3, 3), activation='gelu', padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    #p1 = keras.layers.BatchNormalization()(p1)

    c2 = keras.layers.Conv2D(64, (3, 3), activation='gelu', padding='same')(p1)
    # c2 = keras.layers.Dropout(0.3)(c2)
    c2 = keras.layers.Conv2D(64, (5, 5), activation='gelu', padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    #p2 = keras.layers.BatchNormalization()(p2)

    c3 = keras.layers.Conv2D(64, (3, 3), activation='gelu', padding='same')(p2)
    # c3 = keras.layers.Dropout(0.3)(c3)
    c3 = keras.layers.Conv2D(64, (5, 5), activation='gelu', padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    #p3 = keras.layers.BatchNormalization()(p3)

    
    c4 = keras.layers.Conv2D(128, (3, 3), activation='gelu', padding='same')(p3)
    # c4 = keras.layers.Dropout(0.3)(c4)
    c4 = keras.layers.Conv2D(128, (3, 3), activation='gelu', padding='same')(c4)
    p4 = keras.layers.MaxPooling2D((2, 2))(c4)
    #p4 = keras.layers.BatchNormalization()(p4)

    # bottleneck

    c5 = resnet_block(p4, 256)

    # c5 = keras.layers.Conv2D(256, (3, 3), activation='gelu', padding='same')(p4)
    # c5 = keras.layers.Conv2D(256, (3, 3), activation='gelu', padding='same')(c5)


    #decoder

    u6 = keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation='gelu', padding='same')(u6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation='gelu', padding='same')(c6)

    u7 = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (5, 5), activation='gelu', padding='same')(u7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation='gelu', padding='same')(c7)
    #c7 = keras.layers.BatchNormalization()(c7)

    u8 = keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    #c8 = keras.layers.Conv2D(64, (5, 5), activation='gelu', padding='same')(u8)
    c8 = keras.layers.Conv2D(64, (3, 3), activation='gelu', padding='same')(u8)
    #c8 = keras.layers.BatchNormalization()(c8)

    u9 = keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1])
    c9 = keras.layers.Conv2D(32, (3, 3), activation='gelu', padding='same')(u9)
    #c9 = keras.layers.Conv2D(32, (3, 3), activation='gelu', padding='same')(c9)
    #c9 = keras.layers.BatchNormalization()(c9)

    # output layer
    outputs = keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)

    model = keras.Model(inputs=[inputs], outputs=[outputs])

    return model













# create and compile the model
denoise_model = denoising_unet()
denoise_model.summary()


# Custom SSIM Loss Function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


# custom ssim + l1 loss: (inspired from Cycle-GAN {https://tandon-a.github.io/CycleGAN_ssim/})

def custom_ssim_mae_loss(alpha=0.4):
    def loss(y_true, y_pred):
        ssim_loss = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        mae = tf.keras.losses.MeanAbsoluteError()
        mae_loss = mae(y_true, y_pred)
        return alpha*(1-ssim_loss) + (1-alpha)*mae_loss
    return loss


losses_list = [tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.MeanSquaredError()]
metrics_list = ['mean_absolute_error', 'val_mean_absolute_error', 
                'mean_squared_error', 'val_mean_squared_error']


denoise_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=start_learning_rate,), 
                      loss=custom_ssim_mae_loss(alpha=0.4), metrics=[metrics_list[0]])

# save the best model based on the validation KID metric
checkpoint_path = path_to_dir + "/bgrem_unet/checkpoints/"
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path + "unet_aec_bgrem_best_model_sqrtIm-Asi-Tot-Src-Ch0-SSIM-MAE.weights.h5", 
                                      save_weights_only=True, 
                                      monitor="val_loss", 
                                      mode="min", 
                                      save_best_only=True, 
                                      verbose=1)

def schedule(epoch):
   return(learning_rate_decay**epoch*start_learning_rate)

# learning_rate_scheduler = LearningRateScheduler(schedule, verbose=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.8, patience=3, 
                              min_lr=1e-8, verbose=1)
# check loss values and if doesn't decrease include early stopping, else for now we use 40/60 epochs and don't>

early_stopping = EarlyStopping(monitor="val_loss",  # Monitors the validation image loss
                               patience=10,           # Number of epochs with no improvement after which training will be stopped
                               restore_best_weights=True,  # Restore model weights from the best epoch
                               verbose=1              # Print messages when stopping
                               )

# run training and plot generated images periodically
history = denoise_model.fit(train_image_ds, epochs=num_epochs, 
                            validation_data=val_image_ds, 
                            callbacks=[checkpoint_callback, 
                                       reduce_lr, early_stopping,],)

############################
# Track the loss vs epochs
###########################
def plot_training_history(history):
    # Extract the loss values
    epochs = range(1, len(history.history['loss']) + 1)

    # Plot Noise Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(epochs, history.history['loss'], label='Train: Loss')
    plt.plot(epochs, history.history['val_loss'], label='Valid: Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MAE Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()

    # Plot Image Loss
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(epochs, history.history[metrics_list[0]], label='Train: MAE')
    plt.plot(epochs, history.history[metrics_list[1]], label='Valid: MAE')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('MAE vs. Epochs')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(path_to_dir + 'bgrem_unet/SSIM_MAE_loss_sqrt_vs_epochs-Asi-Tot-Src-Ch0.png', dpi=200)
    plt.show()

plot_training_history(history)




################################
# let's try some predictions
################################

denoise_model.load_weights(checkpoint_path + 'unet_aec_bgrem_best_model_sqrtIm-Asi-Tot-Src-Ch0-SSIM-MAE.weights.h5')

def plot_test_predictions(test_dataset, model, num_samples=3):
    # Make an iterator from the test dataset
    iterator = iter(test_dataset)
    
    # Retrieve the first batch (assuming batch size is at least as large as num_samples)
    ((total_images, source_images), (tot_max_vals, src_max_vals)) = next(iterator)
    # noisy_images, clean_images = total_images[:num_samples], source_images[:num_samples]
    # noisy_max_vals, clean_max_vals = tot_max_vals[:num_samples], src_max_vals[:num_samples]
    
    # Randomly select indices for the images to display
    indices = np.random.choice(range(total_images.shape[0]), size=num_samples, replace=False)
    
    # Predict using the model
    predicted_images = model.predict(total_images)
    # diff_img = np.abs(total_images - predicted_images)


    
    
    # Set up the plot - 4 rows (Total, Source, Predicted) and num_samples columns
    fig, axes = plt.subplots(5, num_samples, figsize=(12, 8))
    
    for i, idx in enumerate(indices):

        tot_im_org = total_images[idx].numpy() * tot_max_vals[idx].numpy()
        src_im_org = source_images[idx].numpy() * src_max_vals[idx].numpy()
        prd_im_org = predicted_images[idx] * src_max_vals[idx].numpy()
        diff_im = np.abs(tot_im_org - prd_im_org)
        residual_im = np.abs(src_im_org - prd_im_org)

        # Plot total image
        ax1 = axes[0, i]
        # im1 = ax1.imshow(np.sqrt(total_images[idx].numpy()[:, :, 0]), cmap='inferno')  # Adjust color map or normalization as needed
        im1 = ax1.imshow(np.sqrt(tot_im_org[:, :, 0]), cmap='inferno')
        ax1.axis('off')  # Hide axes
        ax1.set_title(f'Total Image (X) {idx+1}')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Plot source image
        ax2 = axes[1, i]
        # im2 = ax2.imshow(np.sqrt(source_images[idx].numpy()[:, :, 0]), cmap='inferno')  # Adjust color map or normalization as needed
        im2 = ax2.imshow(np.sqrt(src_im_org[:, :, 0]), cmap='inferno')
        ax2.axis('off')  # Hide axes
        ax2.set_title(f'Final Image (y) {idx+1}')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Plot predicted denoised image
        ax3 = axes[2, i]
        # im3 = ax3.imshow(np.sqrt(diff_img[idx, :, :, 0]), cmap='inferno')  # Adjust color map or normalization as needed
        im3 = ax3.imshow(np.sqrt(prd_im_org[:, :, 0]), cmap='inferno')
        ax3.axis('off')  # Hide axes
        ax3.set_title(f"Predicted Image (y') {idx+1}")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Plot diff image
        ax4 = axes[3, i]
        # im3 = ax3.imshow(np.sqrt(diff_img[idx, :, :, 0]), cmap='inferno')  # Adjust color map or normalization as needed
        im4 = ax4.imshow(np.sqrt(diff_im[:, :, 0]), cmap='inferno')
        ax4.axis('off')  # Hide axes
        ax4.set_title(f"Difference: Abs(X - y') {idx+1}")
        fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5 = axes[4, i]
        im5 = ax5.imshow(np.sqrt(residual_im[:, :, 0]), cmap='inferno')
        ax5.axis('off')
        ax5.set_title(f"Residual: Abs(y - y') {idx+1}")
        fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)


    
    fig.tight_layout()
    plt.savefig(path_to_dir + f'/bgrem_unet/noise_removal_sqrt_SSIM_MAE/predict_ims_sqrt_{idx+1}-Tot-Src-Asi-Ch0.png', bbox_inches='tight', dpi=200)
    plt.show()

# Usage example
plot_test_predictions(test_ds, denoise_model)
