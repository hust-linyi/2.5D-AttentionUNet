import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_block import _ConvIN3D, _ConvINReLU3D


class SpatialAttention_sigmoid(nn.Module):
    def __init__(self, in_channels, kernel_size=3):  # perform 3D spatial attention
        super(SpatialAttention_sigmoid, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels // 2, (self.kernel_size, 1, 1), padding=(self.kernel_size //2, 0, 0)),
            nn.InstanceNorm3d(self.in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.in_channels // 2, 1, (1, self.kernel_size, 1), padding=(0, self.kernel_size // 2,0)),
            nn.InstanceNorm3d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.in_channels, self.in_channels // 2, (1,self.kernel_size, 1), padding=(0,self.kernel_size // 2, 0)),
            nn.InstanceNorm3d(self.in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.in_channels // 2, 1, (self.kernel_size, 1, 1), padding=(self.kernel_size // 2, 0, 0)),
            nn.InstanceNorm3d(1),
        )

    def forward(self, x):
        grp1_feats = self.conv1(x)   # emulate k*k conv to capture local dependency
        grp2_feats = self.conv2(x)
        weight = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        feature_map = weight*x
        return feature_map+x,weight  # residual


class ChannelWiseAttention(nn.Module):  # from SA net
    def __init__(self, in_channels):
        super(ChannelWiseAttention, self).__init__()

        self.in_channels = in_channels
        self.conv = nn.Sequential(
            _ConvINReLU3D(in_channels=self.in_channels,out_channels=self.in_channels // 4,kernel_size=3, padding=1),
            _ConvIN3D(in_channels=self.in_channels // 4,out_channels=self.in_channels,kernel_size=3, padding=1)
        )

    def forward(self, x):
        size = x.size()[2:]
        feats = F.adaptive_avg_pool3d(x, (size[0] //2,size[1]//2,size[2]//2))
        feats = self.conv(feats)
        weight = F.interpolate(feats,size=size,mode="trilinear",align_corners=True)
        feature_map = weight*x
        return feature_map+x

