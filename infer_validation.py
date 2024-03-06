# this file is used to compare the dice between prediction and original label directly
import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import monai
from monai.data import decollate_batch
from monai.metrics import DiceMetric,ConfusionMatrixMetric,compute_confusion_matrix_metric
from process import val_label_post_transform
from utils import save_metric
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    Orientationd,
    EnsureTyped
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transforms(keys=("label","pred")):  # a little unsure about keys
    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
    ]
    xforms.extend([EnsureTyped(keys)])
    return Compose(xforms)


# label folder indicates the directory of ground truth label while pred_folder indicates the prediction
def seg_compare_metric(label_folder,pred_folder,save_dir="report"):
    labels = sorted(glob.glob(os.path.join(label_folder, "*_seg.nii.gz")))
    preds = sorted(glob.glob(os.path.join(pred_folder, "MR*","*_seg.nii.gz")))
    keys = ("label", "pred")
    infer_file = [{keys[0]: seg1, keys[1]: seg2} for seg1, seg2 in zip(labels, preds)]
    infer_ds = monai.data.CacheDataset(data=infer_file,transform=transforms())
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    Precision = ConfusionMatrixMetric(include_background=False, metric_name="precision")
    Recall = ConfusionMatrixMetric(include_background=False, metric_name="recall")
    MeanDice = DiceMetric(include_background=False)
    with torch.no_grad():
        for data in infer_loader:
            label, prediction = data["label"].to(device), data["pred"].to(device)
            label = [val_label_post_transform()(i) for i in decollate_batch(label)]
            prediction = [val_label_post_transform()(i) for i in decollate_batch(prediction)]
            MeanDice(y_pred=prediction, y=label)
            Precision(y_pred=prediction, y=label)
            Recall(y_pred=prediction, y=label)

        mean_dice = MeanDice.aggregate().item()
        precision = (Precision.aggregate()[0]).item()
        recall = (Recall.aggregate()[0]).item()
        mean_dice_raw_data = MeanDice.get_buffer()  # obtain every score in a tensor
        precision_raw_data = compute_confusion_matrix_metric("precision", Precision.get_buffer())
        recall_raw_data = compute_confusion_matrix_metric("recall", Recall.get_buffer())
        metric_detail = {f"mean_dice": mean_dice_raw_data, f"precision": precision_raw_data,
                         f"recall": recall_raw_data}
        print("Mean Dice:", f"{mean_dice:.5f}", "  precision:", f'{precision:.5f}', "  recall:", f'{recall:.5f}')
        save_metric(metric_detail, labels,save_dir)

