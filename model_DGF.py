#cd desktop/denoising
import torch
import torch.nn as nn
from model import *
from guided_filter_pytorch.guided_filter import ConvGuidedFilter
##

class SFD_C_DGF(nn.Module):
    def __init__(self, in_channels = 3, out_channels=3):
        
        super(SFD_C_DGF, self).__init__()
        
        self.layer_SFD= SFD_C(in_channels = in_channels, out_channels=out_channels)
        self.gf = ConvGuidedFilter(radius=1)
        # self.layer4=nn.Conv2d(64,out_channels,3,stride=1,padding=1)
        #self.layer9=nn.BatchNorm2d(1)
    
    def forward(self, x_lr,x_hr):
        out_lr=self.layer_SFD(x_lr)
        out_hr=self.gf(x_lr, out_lr, x_hr)

        #out=self.layer9(out) no BN at the end, no res learning
        #out += x
        return out_hr

##

"""
To initialize the network you need un 8 layers single frame denoiser (in fact 9 because because of the last batchnorm).
The network takes in input the current frame and the features of the last frame of the reccurent part. 

It outputs the denoised images by the single and the multi frame denoiser but also the features of each layer of the muti frame denoiseur.
"""


class MFD_C_DFG(nn.Module):
    def __init__(self,SFD):
        
        super(MFD_C_DFG, self).__init__()
        self.layer_MFD= MFD_C(SFD.layer_SFD)
        self.gf = SFD.gf


    def forward(self, x_lr,x_hr, mf1, mf2,mf3,mf4,mf5,mf6,mf7,mf8):
        out_lr,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8=self.layer_MFD(x_lr, mf1, mf2,mf3,mf4,mf5,mf6,mf7,mf8)
        out_hr = self.gf(x_lr, out_lr, x_hr)
        mf8_hr = self.gf(x_lr, mf8, x_hr)
        return out_lr,out_hr,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8,mf8_hr
