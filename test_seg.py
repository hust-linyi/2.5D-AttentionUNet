import logging
import glob
import os
import sys
from infer_validation import seg_compare_metric
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from validation import get_inferer
import torch
from Unet_seg import Unet
from monai.data import Dataset, DataLoader
from process import get_xforms
from monai.transforms import RandGaussianNoised
import monai

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test(img_size,net,model_folder,output_folder):
    ckpts = []
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    keys = ("image",)
    inferer = get_inferer(patch_size=img_size)
    data_folder = "YOUR_DATA_FOLOER"
    images = sorted(glob.glob(os.path.join(data_folder,"*_seg.nii.gz")))
    model_folder = sorted(glob.glob(os.path.join(model_folder,"-fold*")))
    for i in range (len(model_folder)):
        model_path = sorted(glob.glob(os.path.join(model_folder[i],"*.pth")))
        ckpts.append(model_path[-1])

    infer_transforms = get_xforms("infer", keys)
    saver = monai.data.NiftiSaver(output_dir=output_folder, mode="nearest")
    datalist = [{keys[0]: img} for img in images]
    infer_ds = Dataset(
        data = datalist,
        transform= infer_transforms
    )
    infer_loader = DataLoader(
        dataset=infer_ds,
        num_workers=4,
        batch_size=1,
        pin_memory=torch.cuda.is_available(),
    )

    net.eval()
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
            n = 0.0
            preds = 0.0
            for i in range(len(ckpts)):
                net.load_state_dict(torch.load(ckpts[i], map_location=device))
                preds = preds + inferer(infer_data[keys[0]].to(device), net)
                n = n + 1.0
                for _ in range(4):
                    # test time augmentations
                    _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                    pred = inferer(_img.to(device), net)
                    preds = preds + pred
                    n = n + 1.0
                    for dims in [[2], [3]]:
                        flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                        pred = torch.flip(flip_pred, dims=dims)
                        preds = preds + pred
                        n = n + 1.0
                        
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])
            seg_compare_metric(label_folder="Zhujiang hospital",pred_folder="test_pred")


if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test((192,192,16),Unet(dropout=0.2).to(device),
    model_folder="test_runs",output_folder="test_pred")