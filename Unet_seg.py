import torch
import torch.nn as nn
import torch.nn.functional as F
from context_block import AnisotropicMaxPooling  # , PyramidPooling
from basic_block import _ConvIN3D,_ConvINReLU3D
from Attention import SpatialAttention_sigmoid as SpatialAttention, ChannelWiseAttention
# this model is used to solve isotropic problems


class TwoConv_decoder(nn.Module):  # learn from Efficient Context-Aware Network
    def __init__(self, in_chns,out_chns,dropout):
        super(TwoConv_decoder, self).__init__()
        self.conv1 = nn.Sequential(   # 3x3x3 is implemented in 3x3x1->1x1x3 for anisotropic kernel
             _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=(3,3,1),padding=(1,1,0),p=dropout),
             _ConvIN3D(in_channels=in_chns, out_channels=out_chns, kernel_size=(1,1,3), padding=(0,0,1)),
        )
        self.conv2 = nn.Conv3d(in_channels=in_chns,out_channels=out_chns,kernel_size=1,padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        x1 = self.relu(x1 + x)  # residual block
        return x1


class TwoConv_encoder_1(nn.Module):
    def __init__(self, in_chns,out_chns,dropout):  # the order is implemented from monai.networks.blocks.ADN
        super(TwoConv_encoder_1, self).__init__()
        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns, out_channels=out_chns, kernel_size=(3,3,1), padding=(1,1,0), p=dropout),
            _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=(3,3,1), padding=(1,1,0)),
        )
        self.conv2 = nn.Sequential(
            _ConvINReLU3D(in_channels=out_chns, out_channels=out_chns, kernel_size=(3,3,1), padding=(1,1,0),p=dropout),
            _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=(3,3,1), padding=(1,1,0)),
        )
        self.conv3 = nn.Conv3d(in_channels=in_chns, out_channels=out_chns, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv3(x)
        x1 = self.relu(x1+x)
        x2 = self.conv2(x1)
        x2 = self.relu(x2+x1)
        return x2


class TwoConv_encoder_2(nn.Module):
    def __init__(self, in_chns,out_chns,dropout):
        super(TwoConv_encoder_2, self).__init__()
        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=3,padding=1,p=dropout),
            _ConvIN3D(in_channels=out_chns,out_channels=out_chns,kernel_size=3,padding=1),
        )
        self.conv2 = nn.Sequential(
            _ConvINReLU3D(in_channels=out_chns, out_channels=out_chns, kernel_size=3, padding=1,p=dropout),
            _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=3, padding=1),
        )
        self.conv3 = nn.Conv3d(in_channels=in_chns, out_channels=out_chns, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv3(x)
        x1 = self.relu(x1+x)
        x2 = self.conv2(x1)
        x2 = self.relu(x2+x1)
        return x2


class Down_1(nn.Sequential):
    def __init__(self, in_chns, out_chns, dropout):
        super(Down_1, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)),
            TwoConv_encoder_1(in_chns,out_chns,dropout)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Down_2(nn.Sequential):
    def __init__(self, in_chns, out_chns,dropout):
        super(Down_2, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,1),stride=(2,2,1)),
            TwoConv_encoder_2(in_chns,out_chns,dropout)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Down_3(nn.Sequential):
    def __init__(self, in_chns, out_chns,dropout):
        super(Down_3, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2,stride=2),
            TwoConv_encoder_2(in_chns,out_chns,dropout)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up_1(nn.Module):
    def __init__(self, in_chns, out_chns, attention_block, dropout,halves=True):
        super(Up_1, self).__init__()
        up_chns = in_chns//2 if halves else in_chns
        self.up = nn.ConvTranspose3d(in_chns,up_chns,kernel_size=(2,2,1),stride=(2,2,1))
        self.attention = attention_block
        self.convs = TwoConv_decoder(up_chns ,out_chns,dropout)

    def forward(self,x1,x2):   # x1 is to perform upsampling, x2 is the encoder block
        x_1 = self.up(x1)
        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        if x2 is not None:
            dimensions = len(x1.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x2.shape[-i - 1] != x_1.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_1 = F.pad(x_1, sp, "replicate")
            x = x2+x_1   # concatenation or summation?
            x,w = self.attention(x)
            x = self.convs(x)
            return x,w
        else:
            x = self.convs(x_1)
            return x


class Up_2(nn.Module):
    def __init__(self, in_chns, out_chns, attention_block,dropout,halves=True):
        super(Up_2, self).__init__()
        up_chns = in_chns//2 if halves else in_chns
        self.up = nn.ConvTranspose3d(in_chns,up_chns,kernel_size=2,stride=2)
        self.convs = TwoConv_decoder(up_chns ,out_chns,dropout)
        self.attention = attention_block

    def forward(self,x1,x2): # x1 is to perform upsampling, x2 is the encoder block
        x_1 = self.up(x1)
        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        if x2 is not None:
            dimensions = len(x1.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x2.shape[-i - 1] != x_1.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_1 = F.pad(x_1, sp, "replicate")
            x = x2+x_1
            x = self.attention(x)
            x = self.convs(x)
        else:
            x = self.convs(x_1)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int = 2,
        features: tuple =(32, 32, 64, 128, 256, 32),
        dropout=0.0
    ):
        super().__init__()
        self.w0 = None
        self.w1 = None
        self.w2 = None
        self.conv_0 = TwoConv_encoder_1(in_channels, features[0], dropout)
        self.attention_0 = SpatialAttention(features[0])
        self.down_1 = Down_1(features[0], features[1],dropout)  # 192x192x16->96x96x16
        self.attention_1 = SpatialAttention(features[1])
        self.down_2 = Down_1(features[1], features[2],dropout)  # 96x96x16 ->48x48x16
        self.attention_2 = SpatialAttention(features[2])
        self.down_3 = Down_2(features[2], features[3],dropout)   # 48x48x16-> 24x24x16
        self.attention_3 = ChannelWiseAttention(features[3])
        self.down_4 = Down_3(features[3], features[4],dropout)   # 24x24x16 -> 12x12x8
        self.context_block_1 = AnisotropicMaxPooling(features[4],features[4])
        self.attention_4 = ChannelWiseAttention(features[4])
        self.up_4 = Up_2(features[4], features[3],self.attention_3,dropout)
        self.up_3 = Up_1(features[3], features[2],self.attention_2,dropout)
        self.up_2 = Up_1(features[2], features[1],self.attention_1,dropout)
        self.up_1 = Up_1(features[1], features[5],self.attention_0,dropout,halves=False)
        # self.context_block_2 = PyramidPooling(features[5])
        self.final_conv = nn.Conv3d(features[5] ,out_channels,kernel_size=1,stride=1)

    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        x4 = self.context_block_1(x4)
        _x4 = self.attention_4(x4)
        u4 = self.up_4(_x4, x3)
        u3,w2 = self.up_3(u4, x2)
        u2,w1 = self.up_2(u3, x1)
        u1,w0 = self.up_1(u2, x0)
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        # u0 = self.context_block_2(_u1)
        logits = self.final_conv(u1)
        return logits
    
    def get_attention_weight(self):
        return self.w0, self.w1, self.w2