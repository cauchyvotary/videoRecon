#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse
import h5py
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

out_file = '/home/suoxin/Body/obj1/mask/masks.hdf5'
mask_dir = '/home/suoxin/Body/obj1/mask/'
with h5py.File(out_file, 'w') as f:
    dset = None
    for i in range(30):
        silh_file = mask_dir + str(i) +'.png'
        silh = cv2.imread(silh_file)

        if dset is None:
            dset = f.create_dataset("masks", (30, silh.shape[0], silh.shape[1]), 'b', chunks=True, compression="lzf")
        silh = cv2.cvtColor(silh, cv2.COLOR_BGR2GRAY)
        _, silh = cv2.threshold(silh, 100, 255, cv2.THRESH_BINARY)
        dset[i] = silh/255.0



