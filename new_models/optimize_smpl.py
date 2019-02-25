#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import h5py
import argparse
import numpy as np
import cPickle as pkl
import skimage.io as io
from opendr.renderer import ColoredRenderer, BoundaryRenderer
from opendr.camera import ProjectPoints
from opendr.geometry import VertNormals
import pylab
from lib.frame import FrameData
from util.logger import log
from models.smpl import Smpl
from new_models.pose_model import Model
from tqdm import tqdm
from objective import pose_objective , mask_objective, silh_from_mask
import chumpy as ch
from util import mesh


colors = {
    # colorbline/print/copy safe:
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
}



def fit_frame(frames,pose_model):

    frame_num = len(frames)
    E ={}
    for i, f in enumerate(tqdm(frames)):
        E['mask_{}'.format(i)] = mask_objective(f.points, f.pose_model, f.camera, f.rn_b, f.rn_m)

        ch.minimize(
            E,
            [pose_model.v_pose],
            method='dogleg',
            options={'maxiter': 15, 'e_3': 0.001}
            #callback=get_cb(frames[0], base_smpl, camera, frustum) if display else None
        )


def main():
    consensus_file = '/home/suoxin/Body/data/chenxin/result30/betas.pkl'
    camera_file = '/home/suoxin/Body/data/chenxin/camera.pkl'
    image_file = '/home/suoxin/Body/data/chenxin/image30/image.hdf5'
    pose_file = '/home/suoxin/Body/data/chenxin/result30/reconstructed_poses.hdf5'
    masks_file = '/home/suoxin/Body/data/chenxin/mask30/masks.hdf5'
    out = '/home/suoxin/Body/data/chenxin/per'
    model = '/home/suoxin/Body/videoavatars/vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'

    vt = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_vt.npy')
    ft = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_ft.npy')

    with open(model, 'rb') as fp:
        model_data = pkl.load(fp)

    with open(camera_file, 'rb') as fp:
        camera_data = pkl.load(fp)

    with open(consensus_file, 'rb') as fp:
        consensus_data = pkl.load(fp)

    pose_data = h5py.File(pose_file, 'r')
    poses = pose_data['pose']
    trans = pose_data['trans']
    masks = h5py.File(masks_file, 'r')['masks']
    images = h5py.File(image_file, 'r')['masks']
    frame_num = masks.shape[0]

    base_smpl = Smpl(model_data)
    base_smpl.betas[:] = consensus_data['betas']
    base_smpl.v_personal[:] = consensus_data['v_personal']

    camera1 = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'],
                           f=camera_data['camera_f'], k=camera_data['camera_k'], v=base_smpl)

    frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}
    def create_frame(i):
        base_smpl.trans[:] = trans[i]
        base_smpl.pose[:] = poses[i]

        f = FrameData()

        dd= {'v_origin': base_smpl.r, 'f': base_smpl.f}
        pose_model = Model(dd)

        camera = ProjectPoints(v=pose_model, t=camera1.t, rt=camera1.rt, c=camera1.c, f=camera1.f, k=camera1.k)

        f.mask = masks[i]
        points = silh_from_mask(masks[i].astype(np.uint8))

        rn_b = BoundaryRenderer(camera=camera, frustum=frustum, f= pose_model.f, num_channels=1)
        rn_m = ColoredRenderer(camera=camera, frustum=frustum, f=pose_model.f, vc=np.zeros_like(pose_model), bgcolor=1,num_channels=1)

        pose_model.v_pose.label = 'v_pose'
        E = {}
        #E['lap'] = pose_model.v_pose
        E['mask_{}'.format(i)] = mask_objective(points, pose_model, camera, rn_b, rn_m)
        print('E_show tree', E['mask_{}'.format(i)].show_tree())


        ch.minimize(
            E,
            [pose_model.v_pose],
            method='dogleg',
            options={'maxiter': 15, 'e_3': 0.001}
        )

        return f

    frames = []
    for i in range(1):
        frames.append(create_frame(i))

    # fit frame v_pose
    #fit_frame(frames, pose_model)


main()
