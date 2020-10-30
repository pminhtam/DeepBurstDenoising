#cd desktop/denoising
import torch
import torch.nn as nn

##


class CONV_BN_RELU(nn.Module):
    def __init__(self,in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3):
        super(CONV_BN_RELU, self).__init__()
        
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        return(out)
        

class SFD_C(nn.Module):
    def __init__(self, in_channels = 3, out_channels=3):
        
        super(SFD_C, self).__init__()
        
        self.layer1=CONV_BN_RELU(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer2=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer3=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.layer4=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # self.layer5=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.layer6=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.layer7=CONV_BN_RELU(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        #
        # self.layer8=nn.Conv2d(64,out_channels,3,stride=1,padding=1)
        self.layer4=nn.Conv2d(64,out_channels,3,stride=1,padding=1)
        #self.layer9=nn.BatchNorm2d(1)
    
    def forward(self, x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        # out=self.layer5(out)
        # out=self.layer6(out)
        # out=self.layer7(out)
        # out=self.layer8(out)
        #out=self.layer9(out) no BN at the end, no res learning
        #out += x
        return out

##

"""
To initialize the network you need un 8 layers single frame denoiser (in fact 9 because because of the last batchnorm).
The network takes in input the current frame and the features of the last frame of the reccurent part. 

It outputs the denoised images by the single and the multi frame denoiser but also the features of each layer of the muti frame denoiseur.
"""


class MFD_C(nn.Module):
    def __init__(self,SFD):
        
        super(MFD_C, self).__init__()
        self.S1=SFD.layer1
        self.S2=SFD.layer2
        self.S3=SFD.layer3
        self.S4=SFD.layer4
        # self.S5=SFD.layer5
        # self.S6=SFD.layer6
        # self.S7=SFD.layer7
        # self.S8=SFD.layer8
        #self.S9=SFD.layer9
        
        self.M1=CONV_BN_RELU(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.M2=CONV_BN_RELU(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.M3=CONV_BN_RELU(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)        
        # self.M4=CONV_BN_RELU(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.M5=CONV_BN_RELU(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.M6=CONV_BN_RELU(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.M7=CONV_BN_RELU(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.M8=nn.Conv2d(in_channels=70, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.M4=nn.Conv2d(in_channels=70, out_channels=3, kernel_size=3, stride=1, padding=1)
        #self.M9=nn.BatchNorm2d(3)
        
        nn.init.kaiming_normal_(self.M1.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.M2.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.M3.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.M4.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.M5.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.M6.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.M7.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_normal_(self.M8.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.M4.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, x, mf1, mf2,mf3,mf4):
        out=self.S1(x)
        # print(out.size())
        # print(mf1.size())
        mf1=self.M1(torch.cat([out,mf1],dim=1))
        out=self.S2(out)
        mf2=self.M2(torch.cat([out,mf1,mf2],dim=1))
        out=self.S3(out)
        mf3=self.M3(torch.cat([out,mf2,mf3],dim=1))
        out=self.S4(out)
        mf4=self.M4(torch.cat([out,mf3,mf4],dim=1))
        # out=self.S5(out)
        # mf5=self.M5(torch.cat([out,mf4,mf5],dim=1))
        # out=self.S6(out)
        # mf6=self.M6(torch.cat([out,mf5,mf6],dim=1))
        # out=self.S7(out)
        # mf7=self.M7(torch.cat([out,mf6,mf7],dim=1))
        # out=self.S8(out)
        # #out=self.S9(out)
        # mf8=self.M8(torch.cat([out,mf7,mf8],dim=1))
        #mf8 += x
        #out += x
        
        # return out,mf1,mf2,mf3,mf4,mf5,mf6,mf7,mf8
        return out,mf1,mf2,mf3,mf4