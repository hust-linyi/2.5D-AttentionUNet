import glob
import os
import pandas as pd
import numpy as np
from scipy import stats
filename = sorted(glob.glob(os.path.join("*unet*","radiology*")))
for csv_file in filename:
    data = pd.read_csv(csv_file)
    arr = data["mean"].values
    arr = (arr,)
    res = stats.bootstrap(arr,statistic=np.mean,method="percentile",n_resamples=20,random_state=0)
    print(res.confidence_interval)