import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResnetEncoder(nn.Module):
    def __init__(self,
                 model_name='resnet18',):
        super().__init__()
        if model_name == 'resnet18':
            self.resnet = models.resnet18(True)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(True)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # 14x14 or 16x16
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1) # 28x28 or 32x32
        
    def forward(self,x):
        out = self.features(x)
        out = self.deconv1(F.relu(out))
        out = self.deconv2(F.relu(out))
        return out

    def encode(self, x):
        return self(x)
    
class ResnetRGBNEncoder(nn.Module):
    def __init__(self,
                 model_name='resnet18',conditioning_shape=0):
        super().__init__()
        if model_name == 'resnet18':
            self.resnet = models.resnet18(True)
            # rgbnb
            self.resnet.conv1 = nn.Conv2d(conditioning_shape, 64, 7, 2, 3)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
            self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # 14x14 or 16x16
            self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1) # 28x28 or 32x32
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(True)
            # Vision + Vision_small + NOCS + BG
            self.resnet.conv1 = nn.Conv2d(conditioning_shape, 64, 7, 2, 3)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
            self.deconv1 = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            self.deconv = nn.Sequential(
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(),
                self.deconv1,
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                self.deconv2,
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                self.deconv3,
            )
        
    def forward(self,x):
        out = self.features(x)
        out = self.deconv(out)
        return out

    def encode(self, x):
        return self(x)
    
class ResnetRGBDEncoder(nn.Module):
    def __init__(self,
                 model_name='resnet18',conditioning_shape=0):
        super().__init__()
        if model_name == 'resnet18':
            self.resnet = models.resnet18(True)
            # rgbdb
            # self.resnet.conv1 = nn.Conv2d(4+3, 64, 7, 2, 3)
            self.resnet.conv1 = nn.Conv2d(conditioning_shape, 64, 7, 2, 3)
            # self.resnet.conv1 = nn.Conv2d(4+4+4+3, 64, 7, 2, 3)
            # rgbb
            # self.resnet.conv1 = nn.Conv2d(3+3, 64, 7, 2, 3)
            # db
            # self.resnet.conv1 = nn.Conv2d(1+3, 64, 7, 2, 3)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
            self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # 14x14 or 16x16
            self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1) # 28x28 or 32x32
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(True)
            self.resnet.conv1 = nn.Conv2d(conditioning_shape, 64, 7, 2, 3)
            # self.resnet.conv1 = nn.Conv2d(1+3, 64, 7, 2, 3)
            # self.resnet.conv1 = nn.Conv2d(3+3, 64, 7, 2, 3)
            self.features = nn.Sequential(*list(self.resnet.children())[:-2])
            self.deconv1 = nn.ConvTranspose2d(2048, 512, kernel_size=4, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
            self.deconv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            self.deconv = nn.Sequential(
                nn.BatchNorm2d(2048),
                nn.LeakyReLU(),
                self.deconv1,
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                self.deconv2,
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                self.deconv3,
            )
            
        
    def forward(self,x):
        out = self.features(x)
        out = self.deconv(out)
        # out1 = self.upsample(out)
        # out2 = self.deconv1(F.leaky_relu(out1))
        # out3 = self.deconv2(F.leaky_relu(out2))
        # out4 = self.deconv3(F.leaky_relu(out3))
        
        return out

    def encode(self, x):
        return self(x)