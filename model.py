
import torch 
import torch.nn as nn
import argparse
import math
import numpy as np
import numpy.random as npr
import scipy.misc
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt 


class DownConv(nn.Module):
    
    def __init__(
            self, kernel, in_channels, out_channels
    ):
        super(DownConv, self).__init__()
        padding=kernel//2
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),)
        
    def forward(self, x):
        return self.downconv(x)
        
class Bottleneck(nn.Module):
    
    def __init__(
            self, kernel, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        padding=kernel//2
        self.rfconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
    
    def forward(self, x):
        return x
        
class UpConv(nn.Module):
    
    def __init__(
            self, kernel, in_channels, out_channels, FC
    ):
        super(UpConv, self).__init__()
        padding=kernel//2
        if FC:
            self.upconv = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(),
               nn.Upsample(scale_factor=2),)
        else: 
            self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel,padding=padding)
        
    def forward(self, x):
        return self.upconv(x)




class BaseModel(nn.Module):
    def __init__(
            self, kernel, NF, NC, N_inputs
    ):
        super(BaseModel, self).__init__()
        
        #########################Down Conv Layer##################
        self.down1  = DownConv(kernel, N_inputs, NF)
        self.down2  = DownConv(kernel, NF, NF*2)
        #########################Buttleneck Layers##################
        self.btlncks= Bottleneck(kernel, NF*2, NF*2)
        #########################Up Conv Layer######################
        self.up1    = UpConv(kernel, NF*2, NF, True)
        self.up2    = UpConv(kernel, NF,NC, True)  
        ############################################################
        self.Ful_Conn= UpConv(kernel,NC, NC, False)
        ###########################################################
        
    def forward(self, x):
        self.output1=self.down1(x)
        self.output2=self.down2(self.output1)
        self.output3=self.btlncks(self.output2)
        self.output4=self.up1(self.output3)
        self.output5=self.up2(self.output4)
        self.output6=self.Ful_Conn(self.output5)
        return self.output6

class CustomUNET(nn.Module):
    def __init__(
            self, kernel, NF, NC, num_in_channels):
        super(CustomUNET, self).__init__()

        padding = kernel // 2
        ####################################################
        self.down1 = DownConv(kernel,num_in_channels, NF)
        self.down2 = DownConv(kernel,NF, NF*2)
        ####################################################
        self.btlneck = Bottleneck(kernel,NF*2, NF*2)
        ####################################################
        self.up1 =  UpConv(kernel, NF*2+NF*2, NF,True)
        self.up2 = UpConv(kernel, NF+NF, NC, True)
        ####################################################
        self.Ful_conn = UpConv(kernel, NC+num_in_channels, NC, False)
        ###################################################

    def forward(self, x):
        ############### YOUR CODE GOES HERE ###############
        self.out1 = self.down1(x)
        self.out2 = self.down2(self.out1)
        ##########
        self.out3 = self.btlneck(self.out2)
        #########
        self.out4 = self.up1(torch.cat((self.out2, self.out3),1))
        self.out5 = self.up2(torch.cat((self.out1, self.out4),1))
        #########
        self.out6 = self.Ful_conn(torch.cat((x, self.out5),1))
        return self.out6
    
    
  
    
class ResNet(nn.Module):
    def __init__(
            self, kernel, NF, NC, num_in_channels):
        super(ResNet, self).__init__()

        padding = kernel // 2
        ####################################################
        self.down1 = Res_Block(kernel,num_in_channels, NF)
        self.down2 = Res_Block(kernel,NF, NF*2)
        ####################################################
        self.btlneck = Res_Block(kernel,NF*2, NF*2)
        ####################################################
        self.up1 =  Res_Block(kernel, NF*2, NF)
        self.up2 = Res_Block(kernel, NF, NC)
        ####################################################
        self.Ful_conn = UpConv(kernel, NC, NC, False)
        ###################################################

    def forward(self, x):
        ############### YOUR CODE GOES HERE ###############
        self.out1 = self.down1(x)
        self.out2 = self.down2(self.out1)
        ##########
        self.out3 = self.btlneck(self.out2)
        #########
        self.out4 = self.up1(self.out3)
        self.out5 = self.up2(self.out4)
        #########
        self.out6 = self.Ful_conn(self.out5)
        return self.out6
    
    
class Res_Block(nn.Module):
    def __init__(self, kernel, in_channels, out_channels):
        super(Res_Block, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel, padding=kernel//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                in_channels, out_channels,kernel, padding=kernel//2),
                nn.BatchNorm2d(out_channels),)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out