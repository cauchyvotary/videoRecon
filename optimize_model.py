#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import h5py
import argparse
import numpy as np
import cPickle as pkl
from util import im
from lib.frame import FrameData
from util.logger import log
from models.smpl import Smpl

from objective import pose_objective, face_objective,face_area_objective, depth_objective
import chumpy as ch
from util import mesh
from util_smpl import pose_trans, visible_check, visible_area
from opendr.simple import *


colors = {
    # colorbline/print/copy safe:
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
}

def cb(opt_rn):
    debug = np.array(opt_rn.r)
    im.show(debug, id='pose', waittime=1)


def origin_visible_face(base_smpl, poses, trans, masks, camera_data):

    camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'], f=camera_data['camera_f'], k=camera_data['camera_k'], v=base_smpl)
    frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}
    rn_vis = ColoredRenderer(f=base_smpl.f, frustum=frustum, camera=camera, vc=np.zeros_like(base_smpl),
                             num_channels=1)
    Frame = []
    for i in range(poses.shape[0]):
        f = FrameData()
        f.pose_i = np.array(poses[i], dtype=np.float32)
        f.trans_i = np.array(trans[i], dtype=np.float32)
        mask = masks[i]
        base_smpl.pose[:] = f.pose_i
        base_smpl.trans[:] = f.trans_i
        f.face_length, f.face_id = visible_area(base_smpl, rn_vis, camera)
        f.base_smpl = base_smpl
        f.camera = camera
        J0 = base_smpl.J[0]
        J0 = J0.reshape((3, 1))
        trans_tmp = f.trans_i.reshape((3, 1))

        f.trans = -cv2.Rodrigues(f.pose_i[:3])[0].dot(J0) + J0 + trans_tmp


        Frame.append(f)

    return Frame


def main():
    consensus_file = '/home/suoxin/Body/obj1/result1/betas.pkl'
    camera_file = '/home/suoxin/Body/obj1/camera.pkl'
    image_file = '/home/suoxin/Body/obj1/texture_image/image.hdf5'
    pose_file = '/home/suoxin/Body/obj1/result1/reconstructed_poses.hdf5'
    masks_file = '/home/suoxin/Body/obj1/mask/masks.hdf5'
    out = '/home/suoxin/Body/obj1/result1/opt_smpl.pkl'
    model = '/home/suoxin/Body/videoavatars/vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'

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

    vt = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_vt.npy')
    ft = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_ft.npy')

    base_smpl = Smpl(model_data)
    base_smpl.betas[:] = consensus_data['betas']
    base_smpl.v_personal[:] = consensus_data['v_personal']
    base_frame = origin_visible_face(base_smpl, poses, trans, masks, camera_data)

    '''
    define optimization variable
    '''
    opt_camera_pose = ch.array(np.array(poses[:, :3]).copy())
    opt_trans = ch.array(np.array(trans).copy())
    opt_pose = ch.array(np.array(poses[1, 3:]).copy())

    for i in range(opt_trans.shape[0]):
        opt_trans[i, :] = base_frame[i].trans.r.squeeze()
        opt_camera_pose[i, :] = poses[i, :3]

    opt_smpl = Smpl(model_data)
    opt_smpl.betas[:] = consensus_data['betas']
    opt_smpl.v_personal[:] = consensus_data['v_personal']
    opt_smpl.pose = ch.concatenate((ch.zeros(3), opt_pose))

    opt_frame = []
    for i in range(poses.shape[0]):
        print('log')
        frame = FrameData()
        frame.opt_trans = opt_trans[i]
        frame.opt_camera_pose = opt_camera_pose[i]
        frame.V = (Rodrigues(frame.opt_camera_pose).dot(opt_smpl.T)).T
        frame.V = frame.V + frame.opt_trans.reshape((1, 3)) + np.array([0.1,1.5,-0.2])
        frame.opt_camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'],
                               f=camera_data['camera_f'], k=camera_data['camera_k'], v = frame.V)
        frame.frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}

        frame.rn_opt = ColoredRenderer(f=opt_smpl.f, frustum=frame.frustum, camera=frame.opt_camera, vc=np.zeros_like(frame.V),num_channels=1)
        opt_frame.append(frame)

    for i in range(1):
        E = {}
        frame = opt_frame[i]
        opt_trans[i, :] = opt_trans[i].r
        opt_camera_pose[i, :] = opt_camera_pose[i].r
        opt_pose.label = 'opt_pose'
        opt_trans.label = 'opt_trans'
        opt_camera_pose.label = 'opt_camera_pose'
        opt_face_length, opt_face_id = visible_area(opt_smpl, frame.rn_opt, frame.opt_camera)

        print('visible face number before optimization  ', np.intersect1d(np.unique(base_frame[i].face_id), np.unique(opt_face_id)).shape)

        # make energy function
        #E['pose_{}'.format(i)] = pose_objective(opt_pose, poses[i, 3:])
        #E['face_{}'.format(i)] = face_area_objective(opt_face_length, opt_face_id, base_frame[i].face_length, base_frame[i].face_id)  #traingle area
        E['depth_{}'.format(i)] = depth_objective(base_frame[i].camera, frame.opt_camera, base_frame[i].face_id, opt_face_id, opt_smpl.f)  # traingle area

        '''
        log.info('## Run optimization')
        ch.minimize(
            E,
            [opt_camera_pose[i], opt_trans[i]],
            method='dogleg',
            options={'maxiter': 15, 'e_3': 0.001},
            callback= cb(frame.rn_opt)
        )
        opt_face_length, opt_face_id = visible_area(opt_smpl, frame.rn_opt, frame.opt_camera)
        common_face_id = np.intersect1d(opt_face_id, base_frame[i].face_id)
        print('common_face_id after optimization  ', common_face_id.shape, base_frame[i].face_id.shape)
        '''

        image = np.array(frame.rn_opt.ravel()).copy()
        visibility = frame.rn_opt.visibility_image.ravel()
        visible = np.nonzero(visibility != 4294967295)[0]
        ind1 = np.in1d(visibility[visible], np.intersect1d(np.unique(base_frame[i].face_id), np.unique(opt_face_id)))
        image[:] = 0
        image[visible[ind1]] = 255
        image = image.reshape((512, 512))
        cv2.imshow('win', image)
        cv2.waitKey()

    with open(out, 'wb') as fp:
        pkl.dump({
            'opt_pose': opt_pose.r,
            'opt_camera_pose': opt_camera_pose.r,
            'opt_trans': opt_trans.r,
        }, fp, protocol=2)
    #mesh.write('/home/suoxin/Body/opt_smpl.obj', opt_smpl.r, opt_smpl.f, vt=vt, ft=ft)

main()
