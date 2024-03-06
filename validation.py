from monai.inferers import SlidingWindowInferer as SlidingWindowInferer
from monai.data import DataLoader
from monai.metrics import DiceMetric,ConfusionMatrixMetric,compute_confusion_matrix_metric
from monai.data import decollate_batch
from process import val_pred_post_transform,val_label_post_transform
import torch
import torch.nn.functional as F
from utils import get_pad_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inferer(patch_size=(128,128,16)):
    """returns a sliding window inference instance."""
    sw_batch_size, overlap = 2, 0.5
    # "gaussian": gives less weight to predictions on edges of windows.
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


def validate(img_size,model,val_ds,fold):
    print("========================= perform validation =========================")
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=4,
    )
    Precision = ConfusionMatrixMetric(include_background=False,metric_name="precision")
    Recall = ConfusionMatrixMetric(include_background=False,metric_name="recall")
    MeanDice = DiceMetric(include_background=False)
    model.eval()
    inferer = get_inferer(patch_size=img_size)
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            # padding when images and label size after resampling are not the same.
            b_i, c_i, h_i,w_i, d_i = val_images.size()
            b_l, c_l, h_l,w_l, d_l = val_labels.size()
            max_h = max(h_i,h_l)
            max_w = max(w_i,w_l)
            max_d = max(d_i,d_l)
            image_pad_size = get_pad_size(val_images,max_h,max_w,max_d)
            label_pad_size = get_pad_size(val_labels,max_h,max_w,max_d)
            if image_pad_size.count(0) !=6:
                val_images = F.pad(val_images,image_pad_size,mode= "reflect")
            if label_pad_size.count(0) !=6:
                val_labels = F.pad(val_labels,label_pad_size,mode = "reflect")
            val_outputs = inferer(val_images,model)
            val_outputs = [val_pred_post_transform()(i) for i in decollate_batch(val_outputs)]
            val_labels = [val_label_post_transform()(i) for i in decollate_batch(val_labels)]
            MeanDice(y_pred=val_outputs,y=val_labels)
            Precision(y_pred=val_outputs,y=val_labels)
            Recall(y_pred=val_outputs,y=val_labels)
        mean_dice = MeanDice.aggregate().item()
        precision = (Precision.aggregate()[0]).item()
        recall = (Recall.aggregate()[0]).item()
        mean_dice_raw_data = MeanDice.get_buffer()  # obtain every score in a tensor
        precision_raw_data = compute_confusion_matrix_metric("precision", Precision.get_buffer())
        recall_raw_data = compute_confusion_matrix_metric("recall", Recall.get_buffer())
        metric_detail = {f"mean_dice_fold {fold}": mean_dice_raw_data, f"precision_fold {fold}": precision_raw_data,
                         f"recall_fold {fold}": recall_raw_data}
        print("Mean Dice:",f"{mean_dice:.5f}", "  precision:",f'{precision:.5f}', "  recall:",f'{recall:.5f}')
        MeanDice.reset()
        Precision.reset()
        Recall.reset()

    return mean_dice, precision, recall, metric_detail




