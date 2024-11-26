import numpy as np


from sklearn.model_selection import train_test_split

import tensorflow as tf
# from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  ModelCheckpoint,  EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import backend as K
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
import datareader_Judit
import neural_nets

folder_path = '/d11/CAC/sbhattacharyya/Downloads/JPR_CTA_Check/patches/' 

###############################
# load src and corresponding masks
################################

(im_list, mk_list, 
 im_arr, mk_arr) = datareader_Judit.read_src_msk_npy(folder_path, sqrt_sc=True)
print ('check len of src and msks: ', len(im_list), 
       len(mk_arr))


#############################
## create  a train val split
##############################

(train_im_f, test_im_f, 
 train_ims, test_ims,
 train_msk, test_msk) = train_test_split(im_list, im_arr, mk_arr, 
                                         test_size=0.2, 
                                         random_state=40)


print ('check train lengths:  ', len(train_ims), len(train_msk))
print ('check test lens: ', len(test_ims), len(test_msk))

print ('example train im shape: ', train_ims[0].shape)
print ('example train mk shape: ', train_msk[0].shape)

num_samples = len(train_ims) # how many ims the network is trained on

###################################
# Load the UNET model
# we use the helper neural_nets
###################################
loaded_model = neural_nets.all_models(h=512, w=512, bins=3, activation_f='sigmoid')
unet = loaded_model.unet_model(act='gelu')
unet.summary()
# Total params: 1400033 (5.34 MB)

###################################
## define the loss
## for now go with combo
## switch between bce and combo
###################################

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




########################
# dice coeff
########################
def dice_coeff(y_true, y_pred, smooth=1e-4): 
 intersection = K.sum(y_true * y_pred, axis=(0, 1))
 union = K.sum(y_true, axis=(0, 1)) + K.sum(y_pred, axis=(0, 1))
 dice = (2. * intersection + smooth)/(union + smooth)
 return 1-dice

################################
# prepare for compilation
################################

##############################
# model compile
##############################
epochs=200
batch_size=4 # very low batch_size due to low gpu

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.8, patience=5, 
                              min_lr=1e-6, verbose=1)
# check loss values and if doesn't decrease include early stopping, else for now we use 40/60 epochs and don't>

save_checkpoint = ModelCheckpoint(folder_path + 'CTA_JPR_Patches_%dsamples_%dep_ComboLoss_actGelu.weights.h5'%(num_samples, 
                                                                                                       epochs), 
                                  verbose=1, save_weights_only=True, 
                                  save_best_only=True, 
                                  monitor='val_loss')
es = EarlyStopping(monitor='val_loss', 
                   patience = 20, verbose=1, min_delta=1e-7)

# 'binary_crossentropy', 'dice_coeff', 'mean_absolute_error'

unet.compile(optimizer=Adam(learning_rate=5e-4), 
             loss=combo_loss, 
             metrics=[dice_coeff]) #effect learning rate (try diff values)



#################
# training
#################
history = unet.fit(train_ims, train_msk, epochs=epochs, batch_size=batch_size, 
                       validation_data=(test_ims, test_msk), 
                       callbacks=[save_checkpoint,reduce_lr,es])

############################
# plot loss
############################
train_loss = history.history['loss']
val_loss = history.history['val_loss']

train_dice = history.history['dice_coeff']
val_dice = history.history['val_dice_coeff']

fig = plt.figure(figsize=(8, 5))
fig.add_subplot(121)
plt.plot(np.arange(len(train_loss)), train_loss, label="train loss",ls="--")
plt.plot(np.arange(len(train_loss)), val_loss, label="val loss")
#plt.yscale('log')
plt.legend(fontsize=10)
fig.add_subplot(122)
plt.plot(np.arange(len(train_loss)), train_dice, label="train: 1-dice",ls="--")
plt.plot(np.arange(len(train_loss)), val_dice, label="val: 1-dice")
#plt.yscale('log')
plt.legend(fontsize=10)
plt.savefig(folder_path + "train_loss_JPR_patches_sample%d_combo_gelu.png"%(num_samples), dpi=200)

test_loss, test_acc = unet.evaluate(test_ims, test_msk)
print("Accuracy:", test_acc)


###############################
## version info...
###############################
import matplotlib as mpl
import sklearn
import keras
print ('matplotlib version: ', mpl.__version__) # 3.8.4
print ('tf version: ', tf.__version__) # 2.18.0
print ('sklearn version: ', sklearn.__version__) # 1.4.2
print ('keras version: ', keras.__version__) # 3.6.0
