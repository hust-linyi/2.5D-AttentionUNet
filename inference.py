import torch
import os
import logging
import glob
from validation import get_inferer
import monai
from monai.transforms import RandGaussianNoised


def infer(img_size, model, infer_ds, keys=("image",), model_folder="runs",prediction_folder="output"):

    # Return a new list containing all items from the iterable in ascending order.
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pth")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    infer_ds = infer_ds
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    logging.info(f"infer: image ({infer_loader.__len__()})")
    inferer = get_inferer(patch_size=img_size)
    saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
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

