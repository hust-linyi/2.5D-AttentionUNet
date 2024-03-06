import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvINReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, p=0.2):
        super(_ConvINReLU3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout3d(p=p, inplace=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _ConvIN3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvIN3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_chns, out_chns, k=1, p=0, dropout=0.2):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=k,padding=p,p=dropout),
            _ConvIN3D(in_channels=out_chns,out_channels=out_chns,kernel_size=k,padding=p),
        )
        self.conv2 = nn.Sequential(
             _ConvINReLU3D(in_channels=out_chns, out_channels=out_chns, kernel_size=k, padding=p,p=dropout),
             _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=k, padding=p),
        )
        self.conv3 = nn.Conv3d(in_channels=in_chns, out_channels=out_chns, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv3(x)
        x1 = self.relu(x1+x)
        x2 = self.conv2(x1)
        x2 = self.relu(x2+x1)
        return x2
        # return x1


class Decoder(nn.Module):  # learn from Efficient Context-Aware Network
    def __init__(self, in_chns,out_chns,dropout):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(   # 3x3x3 is implemented in 3x3x1->1x1x3 for anisotropic kernel
             _ConvINReLU3D(in_channels=in_chns,out_channels=out_chns,kernel_size=(3,3,1),padding=(1,1,0),p=dropout),
             _ConvIN3D(in_channels=out_chns, out_channels=out_chns, kernel_size=(1,1,3), padding=(0,0,1)),
        )
        self.conv2 = nn.Conv3d(in_channels=in_chns,out_channels=out_chns,kernel_size=1,padding=0)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        x1 = self.relu(x1 + x)  # residual block
        return x1


class Up_cat(nn.Module):  # kernel and stride is for deconvolution
    def __init__(self, in_chns, cat_chns, out_chns, kernel,stride,dropout,attention_block=None, halves=True):
        super(Up_cat, self).__init__()
        up_chns = in_chns//2 if halves else in_chns
        self.up = nn.ConvTranspose3d(in_chns,up_chns,kernel_size=kernel,stride=stride)
        self.attention = attention_block
        self.convs = Decoder(cat_chns+up_chns ,out_chns,dropout)

    def forward(self,x1,x2):   # x1 is to perform upsampling, x2 is from the encoder block
        x_1 = self.up(x1)
        # handling spatial shapes due to the 2x max pooling with odd edge lengths.
        if x2 is not None:
            dimensions = len(x1.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x2.shape[-i - 1] != x_1.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_1 = F.pad(x_1, sp, "replicate")
            x = torch.cat([x2,x_1],dim=1)   # concatenation or summation?
            if self.attention is not None:   # if there is attention
                x,w = self.attention(x)
                x = self.convs(x)
                return x,w
            else:
                x = self.convs(x)
                return x

        else:
            x = self.convs(x_1)
            return x


class Up_sum(nn.Module):  # kernel and stride is for deconvolution
    def __init__(self, in_chns, out_chns, kernel,stride,dropout,attention_block=None, halves=True):
        super(Up_sum, self).__init__()
        up_chns = in_chns//2 if halves else in_chns
        self.up = nn.ConvTranspose3d(in_chns,up_chns,kernel_size=kernel,stride=stride)
        self.attention = attention_block
        self.convs = Decoder(up_chns ,out_chns,dropout)

    def forward(self,x1,x2):   # x1 is to perform upsampling, x2 is from the encoder block
        x_1 = self.up(x1)
        # handling spatial shapes due to the 2x max pooling with odd edge lengths.
        if x2 is not None:
            dimensions = len(x1.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x2.shape[-i - 1] != x_1.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_1 = F.pad(x_1, sp, "replicate")
            x = x_1+x2  # concatenation or summation?
            if self.attention is not None:   # if there is attention
                x,w = self.attention(x)
                x = self.convs(x)
                return x,w
            else:
                x = self.convs(x)
                return x

        else:
            x = self.convs(x_1)
            return x