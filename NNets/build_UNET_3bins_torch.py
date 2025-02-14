import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


import neural_nets_torch, datareader_Judit

print ('!! Check for GPU !!')
print (torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print (device)


folder_path = '/d11/CAC/sbhattacharyya/Downloads/JPR_CTA_Check/patches/all_patches/' 

###############################
# load src and corresponding masks
################################

(im_list, mk_list, 
 im_arr, mk_arr) = datareader_Judit.read_src_msk_npy(folder_path, sqrt_sc=True, log_sc=False)
print ('check len of src and msks: ', len(im_list), 
       len(mk_arr))


# Check if there are any NaNs in the images or masks
print("NaNs in images:", np.isnan(im_arr).any())
print("NaNs in masks:", np.isnan(mk_arr).any())

#############################
## create  a train val split
##############################

(train_im_f, test_im_f, 
 train_ims, test_ims,
 train_msk, test_msk) = train_test_split(im_list, im_arr, mk_arr, 
                                         test_size=0.15, 
                                         random_state=40)


print ('check train lengths:  ', len(train_ims), len(train_msk))
print ('check test lens: ', len(test_ims), len(test_msk))

print ('example train im shape: ', train_ims[0].shape)
print ('example train mk shape: ', train_msk[0].shape)

num_samples = len(train_ims) # how many ims the network is trained on

###########################
# convert to tensors from numpy arrays
###########################

epochs=200
batch_size=16 # very low batch_size due to low gpu

X_train_tensor = torch.tensor(train_ims, dtype=torch.float32)
y_train_tensor = torch.tensor(train_msk, dtype=torch.float32)

X_val_tensor = torch.tensor(test_ims, dtype=torch.float32)
y_val_tensor = torch.tensor(test_msk, dtype=torch.float32)


print("X_train tensor shape:", X_train_tensor.shape)
# shape here (N, H, W, C), good for tensorflow but not for torch
print("y_train tensor shape:", y_train_tensor.shape)
print("X_val tensor shape:", X_val_tensor.shape)
print("y_val tensor shape:", y_val_tensor.shape)




#########################
# reshape to have the channel first:
#########################

X_train_tensor = X_train_tensor.permute(0, 3, 1, 2).float()
y_train_tensor = y_train_tensor.permute(0, 3, 1, 2).float()

X_val_tensor   = X_val_tensor.permute(0, 3, 1, 2).float()
y_val_tensor   = y_val_tensor.permute(0, 3, 1, 2).float()

print("X_train tensor shape after permute:", X_train_tensor.shape)
# shape here (N, C, H, W), good for torch
print("y_train tensor shape after permute:", y_train_tensor.shape)


# Check for NaNs in PyTorch tensors
print("NaNs in PyTorch training images:", torch.isnan(X_train_tensor).any())
print("NaNs in PyTorch training masks:", torch.isnan(y_train_tensor).any())
print("NaNs in PyTorch validation images:", torch.isnan(X_val_tensor).any())
print("NaNs in PyTorch validation masks:", torch.isnan(y_val_tensor).any())


# Create TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                        shuffle=False)





# Checking the first batch in the train_loader
for inputs, targets in train_loader:
    print("Batch inputs shape:", inputs.shape)
    print("Batch targets shape:", targets.shape)
    print("Batch target max: ", targets.max().item())
    print("Batch target min: ", targets.min().item())
    # print("Sample input data:", inputs[0])  # print first input tensor of the batch
    # print("Sample target data:", targets[0])  # print first target tensor of the batch
    break  # Only inspect the first batch

for inputs, targets in val_loader:
    print("Batch (val) inputs shape:", inputs.shape)
    print("Batch (val) targets shape:", targets.shape)
    print("Batch (val) target max: ", targets.max().item())
    print("Batch (val) target min: ", targets.min().item())
    # print("Sample input data:", inputs[0])  # print first input tensor of the batch
    # print("Sample target data:", targets[0])  # print first target tensor of the batch
    break  # Only inspect the first batch



######## check by randomly plotting 3 images in train_loader

def plot_images_and_masks(loader, num_imgs=3):
    fig, axs = plt.subplots(num_imgs, 2, figsize=(10, 3 * num_imgs))  # Set figure size

    for i in range(num_imgs):
        # Randomly select a batch
        idx = np.random.randint(0, len(loader))
        for batch_idx, (images, masks) in enumerate(loader):
            if batch_idx == idx:
                # Randomly select an image and mask from the batch
                img_idx = np.random.randint(0, images.shape[0])
                image = images[img_idx]
                mask = masks[img_idx]

                # Move channel to the last dimension for plotting
                image = image.permute(1, 2, 0)
                mask = mask.permute(1, 2, 0)

                # Plot image
                axs[i, 0].imshow(image.numpy())
                axs[i, 0].set_title(f'Training Image {img_idx}')
                axs[i, 0].axis('off')

                # Plot mask
                axs[i, 1].imshow(mask.numpy(), cmap='gray')
                axs[i, 1].set_title(f'Training Mask {img_idx}')
                axs[i, 1].axis('off')

                break  # Stop after finding the random batch

    plt.tight_layout()
    plt.savefig(folder_path + '/check_torch_loader_example_im.png', dpi=200)

# Call the function with your training DataLoader
plot_images_and_masks(train_loader)



###################################
# Load the UNET model
# we use the helper neural_nets
###################################
loaded_model = neural_nets_torch.AllModels(h=512, w=512, bins=3, 
                                           activation_f='sigmoid', 
                                           activation_i='gelu', dropout_p=0.2)

##### trying to force assign gpu 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print ('check which device the model is using: ', device)
loaded_model.to(device)

dummy_input = torch.randn(1, 3, 512, 512).to(device)
print(summary(loaded_model, input_data=dummy_input, verbose=1))

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


def combo_loss(y_true, y_pred, alpha=0.3, smooth=1e-4):
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



##########################
# Optim & Loss
##########################
learning_rate = 3e-5

optimizer = torch.optim.Adam(loaded_model.parameters(), lr=learning_rate)
criterion = combo_loss
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, 
                              min_lr=2e-8, patience=5, )



best_val_loss = float('inf')
model_save_path = folder_path + 'torch_unet_JPR_%dSamples_check_SqrtSc.pth'%(num_samples)


#############################
# callbacks
####################
# the part below for saving best model
# is inspired from here: 
# https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/
#####################

def save_checkpoint(model, filepath):
    torch.save(model.state_dict(), filepath)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-8):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.epochs_no_improve = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        if self.epochs_no_improve >= self.patience:
            print('Early stopping')
            return True
        return False

early_stopper = EarlyStopping(patience=20, min_delta=1e-8)



# ########################
# # hooks # debugging
# #######################

# def forward_hook(module, input, output):
#     if torch.isnan(output).any():
#         print(f"NaN detected in {module}")

# for name, module in loaded_model.named_modules():
#     module.register_forward_hook(forward_hook)




# #####################
# # check for a single batch # debugging
# #####################

# loaded_model.train()
# inputs, targets = next(iter(train_loader))  # Get first batch from the train_loader

# # Clip targets to be between 0 and 1
# targets_clipped = torch.clamp(targets, min=0, max=1)

# # Forward pass
# outputs = loaded_model(inputs)
# print('Max output:', outputs.max().item(), 'Min output:', outputs.min().item())
# print('Max targets:', targets_clipped.max().item(), 'Min targets:', targets_clipped.min().item())

# # Compute loss
# loss = criterion(targets_clipped, outputs)
# print('Loss (train):', loss.item())

# # Backward pass and optimization
# optimizer.zero_grad()
# loss.backward()

# # Check gradients
# max_grad = 0
# for name, param in loaded_model.named_parameters():
#     if param.grad is not None:
#         param_norm = param.grad.data.norm(2)
#         max_grad = max(max_grad, param_norm)
#         print(f"Grad norm for {name}: {param_norm}")
#     else:
#         print(f"No grad for {name}")

# print(f"Maximum gradient norm observed: {max_grad}")


# torch.nn.utils.clip_grad_norm_(loaded_model.parameters(), max_norm=1.0)  # Clip gradients
# optimizer.step()

# print("Completed single batch processing in training.")


##############################
# model train
##############################
  
train_losses = []
val_losses = []
train_dice = []
val_dice = []



for epoch in range(epochs):
    loaded_model.train()
    train_loss_accum = []
    train_dice_accum = []
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets =  targets.to(device)
        outputs = loaded_model(inputs)
        ### debugs
        # Clip targets to be between 0 and 1
        # targets_clipped = torch.clamp(targets, min=0, max=1)
        #print('Max output:', outputs.max().item(), 'Min output:', outputs.min().item())
        # print('Max targets:', targets_clipped.max().item(), 'Min output:', targets_clipped.min().item())
        ### 
        loss = criterion(targets, outputs)
        # print ('loss (train): ', loss) # debugging
        dice_score = dice_coeff(targets, outputs)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(loaded_model.parameters(), max_norm=1.0)  
        # Clip gradients to avoid explosion
        optimizer.step()
        train_loss_accum.append(loss.item())
        train_dice_accum.append(dice_score.item())

    avg_train_loss = sum(train_loss_accum) / len(train_loss_accum)
    avg_dice_loss  = sum(train_dice_accum) / len(train_dice_accum)
    train_losses.append(avg_train_loss)
    train_dice.append(avg_dice_loss)
    
    # validation step
    loaded_model.eval()
    val_losses_accum = []
    val_dice_accum = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            # targets_clipped = torch.clamp(targets, min=0, max=1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = loaded_model(inputs)
            val_loss = criterion(targets, outputs)
            val_dice_scores = dice_coeff(targets, outputs)
            ### for debug
            # print('Max output (val):', outputs.max().item(), 'Min output:', outputs.min().item())
            # print('Max targets (val):', targets_clipped.max().item(), 'Min target:', targets_clipped.min().item())
            # print ('loss (val): ', val_loss)
            ###
            val_losses_accum.append(val_loss.item())
            val_dice_accum.append(val_dice_scores.item())

    avg_val_loss = sum(val_losses_accum) / len(val_losses_accum)
    avg_val_dice = sum(val_dice_accum) / len(val_dice_accum)
    val_losses.append(avg_val_loss)
    val_dice.append(avg_val_dice)

    print (f'Epoch {epoch + 1}, Loss: {avg_train_loss}, Val Loss: {avg_val_loss}')
    print ('\n')
    print (f' Train Dice: {avg_dice_loss}, Val Dice: {avg_val_dice}')
    print ('Current LR: ', scheduler.get_last_lr())
    
    # did we reach best validation loss:
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_checkpoint(loaded_model, model_save_path)
        print (f'Checkpoint saved at {model_save_path}')
    scheduler.step(avg_val_loss)
    if early_stopper(avg_val_loss):
        break






#####################
# check loss v epoch
#####################
fig = plt.figure(figsize=(9, 5))
fig.add_subplot(121)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (BCE+Dice)')
plt.legend()
fig.add_subplot(122)
plt.plot(train_dice, label='Training Dice')
plt.plot(val_dice, label='Validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice Coeff')
plt.legend()
plt.savefig(folder_path + '/check_loss_v_epoch_torchS%sPatches_SqrtSc.png'%(num_samples), dpi=200)
# plt.grid(True)
# plt.show()


###############################
## version info...
###############################
import matplotlib as mpl
import sklearn
import torch
print ('matplotlib version: ', mpl.__version__) # 3.9.2
print ('torch version: ', torch.__version__) # 2.5.1
print ('sklearn version: ', sklearn.__version__) # 1.5.2
print ('numpy version: ', np.__version__) # 2.1.3
