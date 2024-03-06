import torch
import torch.nn as nn
from typing import List
from monai.networks import one_hot
import torch.nn as nn

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class DiceLoss(nn.Module):    # copy from monai
    def __init__(self,smooth_nr: float = 1e-5,smooth_dr: float = 1e-5,onehot=True):
        super().__init__()
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.one_hot = onehot

    def forward(self, y_pred: torch.Tensor,target: torch.Tensor):
        n_pred_ch = y_pred.shape[1]
        if self.one_hot == True:
            target = one_hot(target,num_classes=n_pred_ch)
            y_pred = torch.softmax(y_pred,dim=1)
        if target.shape != y_pred.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({y_pred.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(y_pred.shape)).tolist()
        intersection = torch.sum(target * y_pred, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(y_pred, dim=reduce_axis)
        denominator = ground_o + pred_o
        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        f = torch.mean(f)
        return f


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        # self.weight = torch.FloatTensor([0.1, 1.0]).cuda()
        # self.cross_entropy = nn.CrossEntropyLoss(weight=self.weight) # simple weighted cross entropy loss

    def forward(self, y_pred, y_true):
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # the original shape is (B,1,D,H,W)
        y_true_long = torch.squeeze(y_true,dim=1).long()
        cross_entropy = self.cross_entropy(y_pred, y_true_long)
        dice_loss = self.dice(y_pred, y_true)
        return dice_loss + cross_entropy


def attention_loss(weight,y_true,avg = 0.2):  # training attention map directly
    size = weight.size()[2:]
    dice = DiceLoss(onehot=False)
    if weight.size() != y_true.size():
        pool = nn.AdaptiveAvgPool3d(size)
        y_true_squeeze = pool(y_true)
    else:
        y_true_squeeze = y_true
    return avg*dice(weight,y_true_squeeze)


def deep_supervision_loss(x,y_true,avg=0.2):
    in_chn = x.size()[1]
    size = x.size()[2:]
    dice = DiceLoss(onehot=False)
    conv = nn.Conv3d(in_channels=in_chn,out_channels=1,kernel_size=1,stride=1)  # similar to prediction
    pool = nn.AdaptiveAvgPool3d(size)
    x = conv(x)
    y_true_squeeze = pool(y_true)
    return avg * dice(x, y_true_squeeze)