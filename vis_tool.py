#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def draw_correspondence(src_img, target_img, src_points, target_points):
    point_num = src_points.shape[0]
    if point_num > 20:
        sample_index = np.random.permutation(point_num)[0:20]
        src_points_ds = src_points[sample_index]
        target_points_ds = target_points[sample_index]
    else:
        src_points_ds = src_points
        target_points_ds = target_points

    h, w = src_img.shape[0:2]
    whole_img = np.concatenate((src_img, target_img), axis=1)
    for i in range(src_points_ds.shape[0]):
        cv2.line(whole_img, tuple(src_points_ds[i]), tuple(target_points_ds[i] + np.array([w, 0])), (255, 0, 0))

    cv2.imshow('vis', whole_img)
    cv2.waitKey()
