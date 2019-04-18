from matplotlib import pyplot as plt
import numpy as np
import scipy
import flylib as flb
from flylib import util
import pandas as pd

fly = flb.NetFly(1540,rootpath='/home/annie/imager/media/imager/FlyDataD/FlyDB/')
# fly = flb.NetFly(1548,rootpath='/home/annie/work/programming/fly_muscle_data/')
fly.open_signals()


flydf = pd.DataFrame()

flydf['t'] = fly.time
flydf['stimulus'] = np.array(fly.experimental_block)
flydf['amp_diff'] = np.array(fly.left_amp)-np.array(fly.right_amp)

for (key,value) in fly.ca_cam_left_model_fits.items():
    flydf[key+'_left'] = value
for (key,value) in fly.ca_cam_right_model_fits.items():
    flydf[key+'_right'] = value

# flydf['ca_pixel_left'] = fly.ca_cam_left #this is really big (obvs)
print(flydf.head())

print(np.unique(flydf['stimulus']))
print(flydf.columns.values)

#Here is an example of how to filter for rows of a certain column value
filtered_df = flydf.loc[flydf['stimulus']=='cl_blocks, g_x=-1, g_y=0, b_x=-8, b_y=0, ch=True']

print(filtered_df.head())

#Or for a combination of column values
idx = (flydf['stimulus']=='cl_blocks, g_x=-1, g_y=0, b_x=-8, b_y=0, ch=True')& \
        (flydf['amp_diff']>0.1)

double_filtered_df = flydf.loc[idx]

print(double_filtered_df.head())

#Access calcium values for a specific muscle and specific stimulus
pretrial_stripe_fix_b2_right = flydf.loc[
    flydf['stimulus']=='pretrial_stripe_fix',['b2_right']]

print(np.shape(pretrial_stripe_fix_b2_right))

#https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
