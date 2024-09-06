## Introduction
This is the official implementation of the paper "MRI-based Deep Learning for Automatic Segmentation and Classification of Fat Metaplasia in Sacroiliac Joint for Axial Spondyloarthritis".

## Requirements
```
monai==0.9.0
torch==1.10.0
```


## Data
We are not able to provide the data used in the paper. However, you can use your own data to train and test the model.

The training data are put in the following structure:
```
MRI images
|—— Train-001.nii.gz
|—— Train-002.nii.gz
|—— Train-003.nii.gz
|—— ...

Ground Truth
|—— Train-001.nii.gz
|—— Train-002.nii.gz
|—— Train-003.nii.gz
|—— ...
```
The test data are put in the following structure:
```
MRI images
|—— Test-001.nii.gz
|—— Test-002.nii.gz
|—— Test-003.nii.gz
|—— ...

Ground Truth
|—— Test-001.nii.gz
|—— Test-002.nii.gz
|—— Test-003.nii.gz
|—— ...
```

## Using the code
1. Train the model and validate using 5-fold cross validation then directly perform testing:
```
python -m main.py
```
2. Separately perform testing:
```
python -m test_seg.py
```

## Result
The results of different models are stored under ["./result"](/result/).  
The results are stored in the following structure:
```
Results of different models
|—— checkpoint(stores the checkpoint of highest dice score we train in each fold)
|   |—— 0.90_fold0.pth
|   |—— 0.80_fold1.pth
|   |—— ...
|—— model configuration and training parameter
|   If you want to train 2.5D-AttentionUnet, please copy all the .py files into ".code" because we use attention_loss + DiceCE_loss instead of just DiceCE_loss
|   |—— training_parameter.txt
|   |—— model.png
|   |—— ...
|—— test_result
|   |—— Test-001
        |—— Test-001.nii.gz
|   |—— Test-002
        |—— Test-002.nii.gz
|   |—— ...
|   |—— mean_dice_raw.csv (Record the mean dice of the test set)
|   |—— precision_raw.csv
|   |—— recall_raw.csv
|—— val_result
|   |—— val_pred
|       |—— Val-001
|           |—— Val-001.nii.gz
|       |—— Val-002
|           |—— Val-002.nii.gz
|       |—— ...
|   |—— val_report
|       |—— mean_dice_fold 0_raw.csv
|       |—— mean_dice_fold 1_raw.csv
|       |—— ...
|       |—— mean_dice_raw.csv(this file records the mean dice of validation set after performing segmentation)
|       |—— precision_fold 0_raw.csv
|       |—— precision_fold 1_raw.csv
|       |—— ...
|       |—— precision_raw.csv
|       |—— recall_fold 0_raw.csv
|       |—— recall_fold 1_raw.csv
|       |—— ...
|       |—— recall_raw.csv
```

## Citation
Please cite the paper if you use the code.
```bibtex
@article{li2024automatic,
  title={Automatic segmentation of fat metaplasia on sacroiliac joint MRI using deep learning},
  author={Li, Xin and Lin, Yi and Xie, Zhuoyao and Lu, Zixiao and Song, Liwen and Ye, Qiang and Wang, Menghong and Fang, Xiao and He, Yi and Chen, Hao and others},
  journal={Insights into Imaging},
  volume={15},
  number={1},
  pages={93},
  year={2024},
  publisher={Springer}
}
```
