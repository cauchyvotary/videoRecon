#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import cPickle as pkl

camera_data = {
    'camera_t': np.zeros(3),
    'camera_rt': np.zeros(3),
    'camera_f': np.array([512, 512]),
    'camera_c': np.array([512, 512]) / 2.,
    'camera_k': np.zeros(5),
    'width': 512,
    'height': 512,
}


with open('/home/suoxin/Body/obj1/result2/camera.pkl', 'wb') as f:
    pkl.dump(camera_data, f, protocol=2)
