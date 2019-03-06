#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import chumpy as ch
import numpy as np
import scipy.sparse as sp
import cv2

'''
    chumpy notes:
        if you inheritate Ch, note the difference between terms and dterms:
        when you modify terms(dterms) themselves, e.g.
            a = SomeClassInheritateCh(xxx)
            a.terms_1 = xxx
            a.dterms_1 = xxx
        this will both trigger on_change func in the NEXT r request
        but when you modify terms via first get then set, e.g.
            a.terms_1[:] = xxx  # think terms_1 as a numpy or chumpy array
            a.dterms_1[:] = xxx
        terms WILL NEITHER trigger on_changed on the next r request
        but dterms WILL trigger on_changed the next r request 
'''


class Try(ch.Ch):
    terms = 'a', 'b', 'c'
    dterms = 'd'

    def __init__(self, *args, **kwargs):
        self.on_changed(self._dirty_vars)

    def on_changed(self, which):
        print 'onchanged'
        if 'a' in which:
            print 'onchanged a'

        if 'd' in which:
            print 'onchanged d'

    def compute_r(self):
        print 'compute_r'
        return self.a + self.b + self.c + self.d.r

    def compute_dr_wrt(self,wrt):
        if wrt is not self.d:
            return None
        else:
            return 1

class CorrFlowDataObj(ch.Ch):
    terms = 'src_img', 'target_img', 'uv_all'
    dterms = 'corr_flow_all'

    # should check the type and range of image

    def __init__(self, *args, **kwargs):
        self.on_changed(self._dirty_vars)

    def on_changed(self, which):
        #print 'on_changed called'
        if 'target_img' in which:
            #print 'on_changed target_img called'
            sobel_normalizer = cv2.Sobel(np.asarray(np.tile(np.arange(10).reshape(1, 10), (10, 1)), np.float64),
                                         cv2.CV_64F, dx=1, dy=0, ksize=1)[5, 5]
            self.target_img_xdiff = cv2.Sobel(self.target_img, cv2.CV_64F, dx=1, dy=0, ksize=1) / sobel_normalizer
            self.target_img_ydiff = cv2.Sobel(self.target_img, cv2.CV_64F, dx=0, dy=1, ksize=1) / sobel_normalizer

    def compute_r(self):
        # print 'compute_r called'
        target_uv = np.round(self.uv_all + self.corr_flow_all.r).astype(np.int)
        target_uv[:, 0] = np.maximum(np.minimum(target_uv[:, 0], self.target_img.shape[1] - 1), 0)
        target_uv[:, 1] = np.maximum(np.minimum(target_uv[:, 1], self.target_img.shape[0] - 1), 0)
        return self.target_img[target_uv[:, 1], target_uv[:, 0]] - self.src_img[self.uv_all[:, 1], self.uv_all[:, 0]]

    def compute_dr_wrt(self, wrt):
        if wrt is not self.corr_flow_all:
            return None

        if len(self.target_img.shape) == 2:
            channels = 1
        else:
            channels = self.target_img.shape[2]

        target_uv = self.uv_all + self.corr_flow_all.r
        target_uv[:, 0] = np.maximum(np.minimum(target_uv[:, 0], self.target_img.shape[1] - 1 - 1e-3), 0)
        target_uv[:, 1] = np.maximum(np.minimum(target_uv[:, 1], self.target_img.shape[0] - 1 - 1e-3), 0)
        target_uv_upperleft = np.floor(target_uv).astype(np.int)
        alpha_u = (target_uv[:, 0] - target_uv_upperleft[:, 0]).reshape(-1, 1)
        alpha_v = (target_uv[:, 1] - target_uv_upperleft[:, 1]).reshape(-1, 1)

        dx = (1 - alpha_u) * (1 - alpha_v) * self.target_img_xdiff[target_uv_upperleft[:, 1], target_uv_upperleft[:, 0]] + \
             alpha_u * (1 - alpha_v) * self.target_img_xdiff[target_uv_upperleft[:, 1], target_uv_upperleft[:, 0] + 1] + \
             (1 - alpha_u) * alpha_v * self.target_img_xdiff[target_uv_upperleft[:, 1] + 1, target_uv_upperleft[:, 0]] + \
             alpha_u * alpha_v * self.target_img_xdiff[target_uv_upperleft[:, 1] + 1, target_uv_upperleft[:, 0] + 1]
        dy = (1 - alpha_u) * (1 - alpha_v) * self.target_img_ydiff[target_uv_upperleft[:, 1], target_uv_upperleft[:, 0]] + \
             alpha_u * (1 - alpha_v) * self.target_img_ydiff[target_uv_upperleft[:, 1], target_uv_upperleft[:, 0] + 1] + \
             (1 - alpha_u) * alpha_v * self.target_img_ydiff[target_uv_upperleft[:, 1] + 1, target_uv_upperleft[:, 0]] + \
             alpha_u * alpha_v * self.target_img_ydiff[target_uv_upperleft[:, 1] + 1, target_uv_upperleft[:, 0] + 1]

        data = np.hstack((dx.reshape(-1, 1), dy.reshape(-1, 1))).ravel()

        I = np.repeat(np.array(range(0, self.corr_flow_all.shape[0] * channels)), 2)
        J = np.repeat(np.array(range(0, self.corr_flow_all.shape[0] * 2)).reshape(-1, 2), 3, axis=0).ravel()

        result = sp.csc_matrix((data, (I, J)), shape=(self.corr_flow_all.shape[0] * channels, self.corr_flow_all.shape[0] * 2))

        return result


class WarpFlowDataObj(ch.Ch):
    terms = 'uv_all', 'K', 'E_ts', 'target_uv_all', 'height', 'width'  # x_target * E_ts * x_src = 0
    dterms = 'warp_flow_all', 'target_warp_flow_all', 's2t_corr_flow'

    def __init__(self, *args, **kwargs):
        self.on_changed(self._dirty_vars)

    def on_changed(self, which):
        if 'K' in which:
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]

        if 'target_uv_all' in which:
            self.target_mask = -np.ones((self.height, self.width), np.int)
            self.target_mask[self.target_uv_all[:, 1], self.target_uv_all[:, 0]] = range(self.target_uv_all.shape[0])

        # update self.target_mask if possible
        if 's2t_corr_flow' in which:
            target_corr_uv = np.round(self.uv_all + self.s2t_corr_flow.r).astype(np.int)
            target_corr_uv[:, 0] = np.maximum(np.minimum(target_corr_uv[:, 0], self.width - 1), 0)
            target_corr_uv[:, 1] = np.maximum(np.minimum(target_corr_uv[:, 1], self.height - 1), 0)

            target_uv_id = self.target_mask[target_corr_uv[:, 1], target_corr_uv[:, 0]]
            self.valid_rigidity_corr_mask = target_uv_id != -1

            # update target_get_uv_id
            self.target_uv_id = target_uv_id[self.valid_rigidity_corr_mask]

        #print 'warp flow on_changed'
        src_uv = self.uv_all + self.warp_flow_all
        src_uv_valid = src_uv[self.valid_rigidity_corr_mask]
        target_warp_uv = self.target_uv_all + self.target_warp_flow_all.r  # numpy array
        target_warp_uv_valid = target_warp_uv[self.target_uv_id]

        kinv_target_uv = (target_warp_uv_valid - np.array([self.cx, self.cy], dtype=np.float)) / \
                         np.array([self.fx, self.fy], dtype=np.float)
        kinv_target_uv_homo = np.hstack((kinv_target_uv, np.ones((kinv_target_uv.shape[0], 1))))
        t_E = kinv_target_uv_homo.dot(self.E_ts)

        kinv_src_uv = (src_uv_valid - np.array([self.cx, self.cy], dtype=np.float)) / \
                      np.array([self.fx, self.fy], dtype=np.float)
        kinv_src_uv_homo = ch.hstack((kinv_src_uv, np.ones((kinv_src_uv.shape[0], 1))))
        self.tEs = ch.sum(t_E * kinv_src_uv_homo, axis=1)

    def compute_r(self):
        return self.tEs.r

    def compute_dr_wrt(self, wrt):
        if wrt == self.warp_flow_all:
            return self.tEs.dr_wrt(wrt)

        else:
            return None