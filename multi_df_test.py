from matplotlib import pyplot as plt
import numpy as np
import scipy
import flylib as flb
from flylib import util
import pandas as pd

fly_nums = range(1540,1545)
multi_fly_df = util.construct_multi_fly_df(
    fly_nums,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')

print(multi_fly_df.columns.values)
print(np.unique(multi_fly_df['stimulus']))

idx = (multi_fly_df['stimulus']=='cl_blocks, g_x=-1, g_y=0, b_x=-8, b_y=0, ch=True')& \
        ((multi_fly_df['amp_diff']>0.1)&(multi_fly_df['amp_diff']<0.104))

# double_filtered_df = multi_fly_df.loc[idx]
# print(double_filtered_df)
