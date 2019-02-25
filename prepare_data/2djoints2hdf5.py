#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse
import h5py
import json
import numpy as np

from glob import glob
from tqdm import tqdm

out_file = '/home/suoxin/Body/obj1/joints/joints.hdf5'
pose_dir = '/home/suoxin/Body/obj1/joints/'

with h5py.File(out_file, 'w') as f:
    poses_dset = f.create_dataset("keypoints", (30, 42), 'f', chunks=True, compression="lzf")

    for i in range(30):
        pose_file = pose_dir + str(i) + '.png_pose.npz'
        pose = np.array(np.load(pose_file)['pose'][:3,:].T)
        pose1 = np.zeros_like(pose)
        pose1[0, :] = pose[13, :]
        pose1[1, :] = pose[12, :]
        pose1[2, :] = pose[8, :]
        pose1[3, :] = pose[7, :]
        pose1[4, :] = pose[6, :]
        pose1[5, :] = pose[9, :]
        pose1[6, :] = pose[10, :]
        pose1[7, :] = pose[11, :]
        pose1[8, :] = pose[2, :]
        pose1[9, :] = pose[1, :]
        pose1[10, :] = pose[0, :]
        pose1[11, :] = pose[3, :]
        pose1[12, :] = pose[4, :]
        pose1[13, :] = pose[5, :]
        print(i)
        poses_dset[i] = pose1.reshape(1,42)
