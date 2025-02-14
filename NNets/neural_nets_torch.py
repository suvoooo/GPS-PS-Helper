import torch
import torch.nn as nn
import torch.nn.functional as F

####################################
## unet but in torch (1st time)
## check neural_nets.py for tf format 
####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AllModels(nn.Module):
    def __init__(self, h:int, w:int, bins:int, 
                 activation_f:str, activation_i:str, dropout_p:float=0.5):
        super(AllModels, self).__init__()
        # calling the init of the base class nn.Module
        self.h = h
        self.w = w
        self.bins = bins
        self.activation_f = activation_f
        self.activation_i = self.get_act(activation_i)
        self.dropout_p = dropout_p

        # if self.activation_i=='relu':
        #     return activation_i==nn.ReLU()
        # elif self.activation_i=='gelu':
        #     return activation_i=nn.GELU()
        # else:
        #     raise ValueError("Unsupported activation function")  
        
        # encoder block for u-net
        self.encoder = nn.Sequential(self.down_block(self.bins, 32, 3, 3), 
                                     self.down_block(32, 32, 3, 3, apply_droput=False), 
                                     self.down_block(32, 64, 3, 5, apply_droput=False), 
                                     self.down_block(64, 128, 3, 5, apply_droput=False))
        self.bridge = self.down_block(128, 256, 3, 3)
        # self.bridge = self.down_block(128, 128, 3,)

        # decoder block
        self.decoder = nn.Sequential(self.up_block(256, 128, 3, 3), 
                                     self.up_block(128+128, 64, 3, 5, add_conv=False), 
                                     self.up_block(64+64, 32, 3, 3, add_conv=False), 
                                     self.up_block(32+32, 32, 3, 3, add_conv=False), 
                                     self.up_block(32+32, 32, 3, 3, add_conv=False, batch_norm=True))
        
        
        self.final = nn.Conv2d(16 + 16, 1, kernel_size=1)


    def forward(self, x):
        x1 = self.encoder[0](x) # from the seq, choose 0th
        # print ('enc1 and input shapeS:', x1.shape, x.shape)
        x2 = self.encoder[1](x1)
        # print ('enc2 shape:', x2.shape)
        x3 = self.encoder[2](x2)
        # print ('enc3 shape:', x3.shape)
        x4 = self.encoder[3](x3)
        # print ('enc4 shape: ', x4.shape)

        # bridge
        x5 = self.bridge(x4)
        # print ('bridge shape: ', x5.shape)

        # expand
        y5 = self.decoder[0](x5)
        # print("y5 size:", y5.size())  # Debugging line 
        # print("x4 size:", x4.size())  # Debugging line
        y5 = torch.cat([y5, x4], dim=1)
        # print("y5 size after concat:", y5.size()) # 256 here
        # print ("problem here: y5", )
        
        y4 = self.decoder[1](y5)
        y4 = torch.cat([y4, x3], dim=1)
        # print ('problem here: y4', )
        # print("y4 size after concat:", y4.size()) # 128 here
        y4 = self.concat_conv(y4, 3, 64+64)
        # print ('y4 device after concat_conv: ', y4.device)
        


        y3 = self.decoder[2](y4)
        y3 = torch.cat([y3, x2], dim=1)
        # print("y3 size after concat:", y3.size())
        y3  = self.concat_conv(y3, 5, 32+32)

        y2 = self.decoder[3](y3)
        y2 = torch.cat([y2, x1], dim=1)
        y2 = self.concat_conv(y2, 5, 32+32, batch_norm=True)
        # print("y2 size after concat:", y2.size())

        y1 = self.decoder[4](y2)

        # print("y1 size :", y1.size())

        f = self.final(y1)
        if self.activation_f=='sigmoid':
            # print ('using sigmoid activation')
            return torch.sigmoid(f)
        elif self.activation_f=='softmax':
            return F.softmax(f, dim=1)
        else:
            return f
    
    
        

    def down_block(self, in_ch, out_ch, kernel_s1, kernel_s2, apply_droput=False):
        # layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_s, padding='same'), 
        #           self.activation_i, nn.BatchNorm2d(out_ch, eps=1e-6),]   
        # batchnorm messes up things
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_s1, padding='same'), 
                  self.activation_i,]      

        if apply_droput:
            layers.append(nn.Dropout(p=self.dropout_p))

        # layers.extend([nn.Conv2d(out_ch, out_ch, kernel_size=kernel_s, padding='same'), 
        #                self.activation_i, nn.BatchNorm2d(out_ch, eps=1e-6), nn.MaxPool2d(kernel_size=2, stride=2)])
        
        layers.extend([nn.Conv2d(out_ch, out_ch, kernel_size=kernel_s2, padding='same'), 
                       self.activation_i, nn.MaxPool2d(kernel_size=2, stride=2)])

        return nn.Sequential(*layers)    
    
    # def up_block(self, in_ch, out_ch, kernel_size):
    #     block = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 
    #                                              kernel_size=kernel_size, 
    #                                              stride=2, 
    #                                              padding=1, 
    #                                              output_padding=1), 
    #                           self.activation_i, 
    #                           nn.BatchNorm2d(out_ch, eps=1e-6))
    #     return block
    
    def up_block(self, in_ch, out_ch, kernel_s1, kernel_s2, add_conv=False, batch_norm=False):
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_s1, 
                                     stride=2, padding=1, output_padding=1), 
                                     self.activation_i, ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        
        if add_conv:
            # layers.extend([nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding='same'), 
            #                self.activation_i, nn.BatchNorm2d(out_ch, eps=1e-6)])
            layers.extend([nn.Conv2d(out_ch, out_ch, kernel_size=kernel_s2, padding='same'), 
                           self.activation_i,])

        return nn.Sequential(*layers)    
    
    def concat_conv(self, x, ks, out_ch, batch_norm=False):
        # layers = nn.Sequential(nn.Conv2d(x.size(1), out_ch, kernel_size=3, padding='same'), 
        #                      self.activation_i, nn.BatchNorm2d(out_ch))
        layers = nn.Sequential(nn.Conv2d(x.size(1), out_ch, kernel_size=ks, padding='same'), 
                               self.activation_i, )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))

        layers = nn.Sequential(*layers)    
        layers = layers.to(x.device)
        
        return layers(x)
    
        
    
    def get_act(self, act):
        if act == 'relu':
            return nn.ReLU()
        elif act == 'gelu':
            return nn.GELU()
        else:
            raise ValueError("Unsupported activation function")
    


    




################################
## things to check
################################
# netowrk hyperparameters
# why having droput produces bad results
# learning rate and loss (currently a mixture of dice and bce loss)
# effects of varying filter sizes (currently only (3, 3) and (5, 5) kernels)

