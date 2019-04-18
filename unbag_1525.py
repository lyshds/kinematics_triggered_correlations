import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import neo
from neo import AxonIO
from cv_bridge import CvBridge, CvBridgeError
from thllib import flylib as flb
import rosbag

fly = flb.NetFly(1525)

abfreader = AxonIO(fly.abfpaths[0])
abffile = abfreader.read()
block = abffile[0]
segment = block.segments[0]
analogsignals = segment.analogsignals
times = np.array(analogsignals[0].times)

signal_idxs = {'abf_electrode':0,
             'abf_wba_left_amp':1,
             'abf_wba_right_amp':2,
             'abf_sync':3,
             'abf_freq':4,
             'abf_left_hutchen':5,
             'abf_right_hutchen':6,
             'abf_led_pulse':7,
             'abf_kinefly_lmr':8,
             'abf_kinefly_left':9,
             'abf_kinefly_right':10,
             'abf_cam_trig':11}

signal_dict = {}
for key,idx in signal_idxs.items():
    signal = analogsignals[idx]
    #print key
    fly.save_hdf5(np.array(signal),key)
    fly.save_txt(str(signal.units).split()[-1],key+'_units')

fly.save_hdf5(times,'abf_times')

inbag = rosbag.Bag(fly.bagpaths[0])

img_msgs = [(topic,msg,t) for topic,msg,t in inbag.read_messages(topics = ['ca_camera_right/image_raw'])]

ca_cam_right_tstamps = list()
ca_cam_right_ros_tstamps = list()
ca_cam_right_imgs = list()

cv_bridge = CvBridge()
for i in range(len(img_msgs)):
    if img_msgs[i][0] == 'ca_camera_right/image_raw':
        ca_cam_right_ros_tstamps.append(float(img_msgs[i][2].to_time()))
        ca_cam_right_tstamps.append(float(img_msgs[i][1].header.stamp.to_time()))
        ca_cam_right_imgs.append(cv_bridge.imgmsg_to_cv2(img_msgs[i][1]))

fly.save_hdf5(np.array(ca_cam_right_imgs),'ca_camera_left')

#ca_cam_right_tstamps is the timestamp recorded on the camera
#ca_cam_right_ros_tstamps is the timestamp recorded when the image
#is picked up on the ros system
#np.std(np.diff(ca_cam_right_ros_tstamps))
#np.std(np.diff(ca_cam_right_tstamps))

ca_cam_right_times = np.array(ca_cam_right_tstamps)
ca_cam_right_times -= ca_cam_right_times[0]

fly.save_hdf5(ca_cam_right_times,'ca_camera_left_times')