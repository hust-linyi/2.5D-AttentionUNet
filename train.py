import os
import monai
import torch
from loss import DiceCELoss,attention_loss as att 
from torch.utils.tensorboard import SummaryWriter
import logging
from validation import validate
from utils import check_model_number,save_metric
import time


def train(fold,img_size,save_dir, model,train_ds,val_ds,model_folder,batch_size=2,max_epochs=300,lr=1e-3, val_interval = 2):
    filelist = [filename['image'] for filename in val_ds.data]
    best_mean_dice = -1
    best_metric_epoch = -1
    writer = SummaryWriter('plot')   # it is used to record to tensorboard
    loss_fn = DiceCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3800,gamma=0.5)
    logging.info(f"epochs {max_epochs}, lr {lr}")

    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=4,
	    shuffle = True,
        pin_memory=torch.cuda.is_available(),
    )
    for epoch in range(max_epochs):
        print("Epoch:",epoch+1)
        time_start = time.time()
        model.train()
        epoch_loss =0.0
        step = 0
        for batch_data in train_loader:
            step +=1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            outputs = model(inputs)
            # w0,w1,w2 = model.get_attention_weight()
            optimizer.zero_grad()
            loss = loss_fn(outputs,labels)
            # DiceCE_loss = loss_fn(outputs,labels)
            # loss = DiceCE_loss+ att(w1,labels)+att(w2,labels)+att(w0,labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"iteration: {step}/{epoch_len}, train_loss: {loss.item():.4f},dice_loss: {loss:.4f}")
            writer.add_scalar(f"train_loss_fold{fold}", loss.item(), epoch_len * epoch + step)  # record loss in tensorboard

        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar(f"average_loss_fold{fold}", epoch_loss, epoch + 1)
        time_end = time.time()
        print(f"training time for epoch {epoch +1 }:",f'{time_end-time_start:.0f}s')
        if (epoch + 1) % val_interval == 0:
            mean_dice, precision, recall, metric_detail = validate(img_size,model,val_ds,fold=fold)
            torch.save(model.state_dict(), os.path.join(model_folder, f"{mean_dice:.4f}_{epoch + 1}.pth"))
            check_model_number(model_folder)
            if mean_dice > best_mean_dice:    # record best metric and its parameter
                best_mean_dice = mean_dice
                best_metric_epoch = epoch + 1  # in this way we know which epoch do we train the max mean_dice
                print("saved new best metric model")
                save_metric(metric_detail,filelist,save_dir=save_dir)  # which is equivalent to Metrics Saver in monai
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, mean_dice, best_mean_dice, best_metric_epoch
                )
            )
         
            writer.add_scalar(f"val_mean_dice_fold{fold}", mean_dice, epoch + 1)   # tensorboard add mean_dice, recall and precision
            writer.add_scalar(f"precision_fold{fold}", precision, epoch + 1)
            writer.add_scalar(f"recall_fold{fold}", recall, epoch + 1)
        print()

    print(f"train completed, best_metric: {best_mean_dice:.4f} at epoch: {best_metric_epoch}")
    writer.close()

