import os
import glob
from monai.handlers.utils import write_metrics_reports
from numpy import genfromtxt
import numpy as np


def check_model_number(model_folder, best_model_number =3):  # keep three best models
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pth")))
    current_model_number = len(ckpts)
    if current_model_number > best_model_number:
        os.remove(ckpts[0])   # delete the model that has the lowest mean_dice


def save_metric(metric_detail, filelist, save_dir="report_2"):
    write_metrics_reports(
        save_dir=save_dir,
        images= filelist,
        metrics= None,
        metric_details=metric_detail,
        summary_ops=None
    )


def get_pad_size(input,h,w,d):   # 解决了y_pred != y_truth 的问题: 在两边reflect pad
    pad_size = []
    length = len(input.size())
    for k in range(length-1,1,-1):
        if k==4:
            diff = max(d-input.size()[k],0)
        elif k==3:
            diff = max(h-input.size()[k],0)
        else:
            diff = max(w-input.size()[k],0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    
    return pad_size


# this is used in classification to delete images when some of the images do not have
# labels in classification because their age is greater than 45
def sift_data(data_folder,label_folder):
    image = []
    label_meta_data = genfromtxt(label_folder,dtype=str,skip_header=1,delimiter=",",usecols=(0,))
    for i in range(len(label_meta_data)):
        image.append(os.path.join(data_folder,label_meta_data[i]+ np.str("_MRI.nii.gz")))
    image = sorted(image)
    return image
