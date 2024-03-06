# this file contains preprocess and post-process
from monai.transforms import (
    AddChanneld,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    NormalizeIntensityd,
    Spacingd,
    SpatialPadd,
    EnsureTyped,
    EnsureType,
    AsDiscrete,
)
import torch
import numpy as np
import monai

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""
    # 使用字典变换时，必须指明该变换是对image做，还是label做。如，LoadImaged（keys='image'）,表明只加载image
    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.0, 1.0, 7.0), mode=("bilinear", "nearest")[: len(keys)]),
        SpatialPadd(keys, spatial_size=(192, 192, 16), mode="reflect"), 
        NormalizeIntensityd(keys[0],channel_wise=True),
    ]
    # 这些裁剪方式都要求数据格式为通道优先格式（必须有通道维度），也就是说要放在 AddChanneld后面使用
    if mode == "train":
        xforms.extend(
            [
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16),num_samples=3),
                SpatialPadd(keys, spatial_size=(192, 192, 16), mode="reflect"), 
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),  
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.float32)
    if mode == "val":        
        dtype = (np.float32, np.float32)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
    return monai.transforms.Compose(xforms)


def val_pred_post_transform():
    return monai.transforms.Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True, to_onehot=2)
        ]
    )


def val_label_post_transform():
    return monai.transforms.Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=False, to_onehot=2)
        ]
    )