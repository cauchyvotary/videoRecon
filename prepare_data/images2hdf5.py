#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import argparse
import h5py
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

"""

This script stores image from a directory in a compressed hdf5 file.

Example:
$ python images2hdf5.py dataset/subject/ images.hdf5

"""

parser = argparse.ArgumentParser()
parser.add_argument('src_folder', type=str)
parser.add_argument('target', type=str)

args = parser.parse_args()

out_file = args.target
image_dir = args.src_folder
image_files = sorted(glob(os.path.join(image_dir, '*.png')) + glob(os.path.join(image_dir, '*.jpg')))

with h5py.File(out_file, 'w') as f:
    dset = None

    for i, color_file in enumerate(tqdm(image_files)):
        color = cv2.imread(color_file, cv2.IMREAD_COLOR)

        if dset is None:
            dset = f.create_dataset("color", (len(image_files), color.shape[0], color.shape[1], color.shape[2]), 'u8', chunks=True, compression="lzf")

        dset[i] = color
