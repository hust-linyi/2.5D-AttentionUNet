import glob
import logging
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from test_seg import test
import monai
import basedataset
from process import get_xforms
from monai.apps import CrossValidation
from train import train
from infer_validation import seg_compare_metric
from inference import infer
from Unet_seg import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

monai.utils.set_determinism(seed=0)
data_folder = "YOUR_DAATA_FOLDER"
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

images = sorted(glob.glob(os.path.join(data_folder, "*_MRI.nii.gz")))
labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
logging.info(f"training: image/label ({len(images)}) folder: {data_folder}")
amp = True
folds = list(range(4))
img_size = (192,192,16)  # patch_size for cropping
save_dir = "report"   # the directory of csv files recording metrics of a particular image
keys = ("image", "label")   # keys for monai transform
datalist = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]
val_transforms = get_xforms("val", keys)
train_transforms = get_xforms("train", keys)
infer_transforms = get_xforms("infer", ("image",))
cvdataset = CrossValidation(
    dataset_cls=basedataset.CVDataset,
    data=datalist,
    nfolds=4,
    seed=12345,
    transform=train_transforms,
) 
for i in range(4):  # perform cross validation
    train_ds = cvdataset.get_dataset(folds=folds[0: i] + folds[(i + 1):])
    val_ds = cvdataset.get_dataset(folds=i, transform=val_transforms)
    model = Unet(dropout=0.2).to(device) 
    # model = get_net().to(device) 
    train(fold=i, img_size=img_size, save_dir=save_dir,model=model, train_ds=train_ds, val_ds=val_ds,
        model_folder=os.path.join("runs", "-fold " + str(i)), max_epochs=400,lr=5e-4)  # train epochs and initial learning rate
    infer(img_size, model, infer_ds=cvdataset.get_dataset(folds=i, transform=infer_transforms),
        model_folder=os.path.join("runs", "-fold " + str(i)))


seg_compare_metric(label_folder=os.path.join("nii_dataset", "Train"), pred_folder="output",save_dir=save_dir)
test(img_size = img_size, net = model,model_folder="runs",output_folder="test_pred")


