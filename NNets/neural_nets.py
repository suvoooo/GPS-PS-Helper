from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Concatenate,Conv2DTranspose, BatchNormalization, ReLU, concatenate,Dropout
from tensorflow.keras.activations import relu, gelu, linear

######################################
# unet model
######################################


class all_models():
  def __init__(self, h:int, w:int, bins:int, activation_f:str):
    '''
    image params + final layer activation
    if original map; we use sigmoid
    if sqrt(map); we use linear activation 
    '''
    self.h = h
    self.w = w
    self.bins = bins
    self.activation_f = activation_f

  def activation_i(self, act=None):
    if act== 'relu':
      return relu  
    elif act== 'gelu' :
      return gelu
    else: 
      return linear

  def unet_model(self, act=None):

    inputs = Input(shape=(self.h, self.w, self.bins))

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(c1)
    b1 = BatchNormalization()(c1)
    # r1 = ReLU()(b1)
    p1 = MaxPooling2D((2, 2))(b1)

    c2 = Conv2D(32, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(c2)
    b2 = BatchNormalization()(c2)
    # r2 = ReLU()(b2)
    p2 = MaxPooling2D((2, 2))(b2)
    
    c3 = Conv2D(64, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(c3)
    b3 = BatchNormalization()(c3)
    # r3 = ReLU()(b3)
    p3 = MaxPooling2D((2, 2))(b3)
    
    c4 = Conv2D(128, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(c4)
    b4 = BatchNormalization()(c4)
    # r4 = ReLU()(b4)
    p4 = MaxPooling2D(pool_size=(2, 2))(b4)

    #bridge 
    c5 = Conv2D(256, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(p4)
    b5 = BatchNormalization()(c5)
    # r5 = ReLU()(b5)
    c5 = Dropout(0.2)(b5)
    c5 = Conv2D(256, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), activation=self.activation_i(act), 
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = BatchNormalization()(u6)
    # u6 = ReLU()(u6)
    
    u7 = Conv2DTranspose(64, (2, 2), activation=self.activation_i(act), 
                         strides=(2, 2), padding='same')(u6)
    u7 = concatenate([u7, c3])
    c6 = Conv2D(64, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(u7)
    u7 = BatchNormalization()(c6)
    # u7 = ReLU()(u7)

    u8 = Conv2DTranspose(32, (2, 2), activation=self.activation_i(act), 
                         strides=(2, 2), padding='same')(u7)
    u8 = concatenate([u8, c2])
    c7 = Conv2D(64, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(u8)
    u8 = BatchNormalization()(c7)
    # u8 = ReLU()(u8)
    
    u9 = Conv2DTranspose(16, (2, 2), activation=self.activation_i(act), 
                         strides=(2, 2), padding='same')(u8)
    u9 = concatenate([u9, c1], axis=3)
    c8 = Conv2D(64, (3, 3), activation=self.activation_i(act), 
                kernel_initializer='he_normal', padding='same')(u9)
    u9 = BatchNormalization()(c8)
    # u9 = ReLU()(u9)
    
    outputs = Conv2D(1, (1, 1), activation=self.activation_f)(u9)
    #outputs = Conv2D(2, (1, 1), activation='softmax')(u9)
    model = Model(inputs=[inputs], outputs=[outputs])

    return model
  
  # add your models with def
  
