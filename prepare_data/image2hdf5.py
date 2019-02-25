#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import h5py
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm


def video2image():
    masks = h5py.File('/home/suoxin/Body/people_snapshot_public/male-2-sport/masks.hdf5', 'r')['masks']
    cap = cv2.VideoCapture('/home/suoxin/Body/people_snapshot_public/male-2-sport/male-2-sport.mp4')
    i=0
    for i in range(masks.shape[0]):
        ret, frame = cap.read()
        if(frame.shape[0]<512):
            break
        image = (frame * masks[i].reshape((frame.shape[0],frame.shape[1],1))).astype(np.uint8)
        cv2.imwrite('/home/suoxin/Body/people_snapshot_public/male-2-sport/image/'+str(i)+'.png', image)
        i=i+1
#video2image()


def sample_image():
    path = '/home/suoxin/Body/people_snapshot_public/male-2-sport/'
    for i in range(80):
        image = cv2.imread(path+ 'warp_image_ray/' + str(i*2) + '.png')
        cv2.imwrite(path + 'image_warp_sample/' + str(i) + '.png',image)

sample_image()

'''
out_file = '/home/suoxin/Body/people_snapshot_public/male-2-sport/image/image.hdf5'
image_dir = '/home/suoxin/Body/people_snapshot_public/male-2-sport/image/'

with h5py.File(out_file, 'w') as f:
    dset = None

    for i in range(467):
        image = cv2.imread(image_dir + str(i) + '.png')

        if dset is None:
            dset = f.create_dataset("masks", (467, image.shape[0], image.shape[1],3), 'b', chunks=True, compression="lzf")
        dset[i] = image/255.0
'''

