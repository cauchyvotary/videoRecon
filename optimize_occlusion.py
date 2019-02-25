#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import h5py
import numpy as np
import cPickle as pkl
from models.smpl import Smpl

#from objective import pose_objective, face_objective
import chumpy as ch

from opendr.camera import ProjectPoints
from opendr.renderer import DepthRenderer, ColoredRenderer
from opendr.geometry import Rodrigues, 
from util import mesh

# poses n*23
# return a average body pose
def average_body(poses):
    return np.average(poses[:,3:], axis=0)

# find the visible pixel in the warp image but invisible in the origin mask

def find_occlusion(rn_opt, rn_base, f):
    visivility_opt = rn_opt.visibility_image.ravel()
    visible_opt = np.nonzero(visivility_opt != 4294967295)[0]
    opt_face_ind = visivility_opt[visible_opt]

    visivility_base = rn_base.visibility_image.ravel()
    visible_base = np.nonzero(visivility_base != 4294967295)[0]
    base_face_ind = visivility_base[visible_base]

    concensus_ind = np.intersect1d(np.unique(opt_face_ind), np.unique(base_face_ind))
    diff_ind = np.setdiff1d(np.unique(opt_face_ind), concensus_ind) # this visible face index in the optimal model but invisible in the original model
    print('diff_ind', diff_ind.shape)
    if(diff_ind.shape[0]==0):
        return

    rn_opt.set(f=f[diff_ind])
    cv2.imshow('win', rn_opt.r)
    cv2.waitKey()
    rn_opt.set(f=f)



def main():

    base_path = '/home/suoxin/Body/obj/'
    consensus_file = base_path + 'result/betas.pkl'

    camera_file = base_path + '/result/camera.pkl'
    image_file = base_path + '/image/image.hdf5'
    pose_file = base_path + '/result/reconstructed_poses.hdf5'
    masks_file = base_path + '/mask/masks.hdf5'
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
    print('poses', poses.shape)
    masks = h5py.File(masks_file, 'r')['masks']

    vt = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_vt.npy')
    ft = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_ft.npy')

    opt_pose = poses[1][3:]
    #opt_pose = average_body(poses)  # get the average body pose

    opt_camera_pose = np.zeros((poses.shape[0],3))

    base_smpl = Smpl(model_data)
    base_smpl.betas[:] = consensus_data['betas']
    base_smpl.v_personal[:] = consensus_data['v_personal']

    camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'],
                           f=camera_data['camera_f'], k=camera_data['camera_k'], v=base_smpl)

    frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}

    rn_base = ColoredRenderer(camera=camera, frustum=frustum, f=base_smpl.f, vc=np.zeros_like(base_smpl), bgcolor=1, num_channels=1)
    drn = DepthRenderer(camera=camera, v=base_smpl, f=base_smpl.f, frustum=frustum)

    opt_smpl = Smpl(model_data)
    opt_smpl.betas[:] = consensus_data['betas']
    opt_smpl.v_personal[:] = consensus_data['v_personal']

    opt_trans_tmp = ch.zeros(3)
    opt_camera_pose_tmp = ch.zeros(3)

    opt_smpl.pose = ch.concatenate((ch.zeros(3), opt_pose))
    opt_V = opt_trans_tmp.reshape((1, 3)) + (Rodrigues(opt_camera_pose_tmp).dot(opt_smpl.T)).T

    opt_camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'], f=camera_data['camera_f'], k=camera_data['camera_k'], v=opt_V)
    rn_opt = ColoredRenderer(camera=opt_camera, frustum=frustum, f=opt_smpl.f, vc=np.zeros_like(opt_smpl), bgcolor=1, num_channels=1)
    drn_opt = DepthRenderer(camera=opt_camera, f=base_smpl.f, frustum=frustum)

    track_camera = ProjectPoints(t=np.zeros(3),rt=np.zeros(3), c=camera_data['camera_c'], f = camera_data['camera_f'], k=camera_data['camera_k'])

    print('camera', camera_data)


    for i in range(0,30):

        base_smpl.pose[:] = poses[i]
        base_smpl.trans[:] = trans[i]

        # obtain the model visible face and the every vertice projection image location
        #base_image_coords, base_visible, base_face_ind, base_smpl_v3d = visible_face_image(rn_base, camera, base_smpl.f)

        J0 = base_smpl.J[0]
        J0 = J0.reshape((3, 1))
        trans_i = np.array(trans[i].copy())
        pose_i = np.array(poses[i, :3].copy())

        trans_tmp = trans_i.reshape((3, 1))
        trans_tmp = cv2.Rodrigues(pose_i[:3])[0].dot(J0) - J0 - trans_tmp
        trans_tmp = - trans_tmp.reshape((1, 3))

        opt_trans_tmp[:] = trans_tmp.r
        opt_camera_pose_tmp[:] = pose_i

        #opt_image_coords, opt_visible, opt_face_ind, opt_smpl_v3d = visible_face_image(rn_opt, opt_camera, opt_smpl.f)
        depth_image_opt = drn_opt.r.copy()
        depth_image_opt[depth_image_opt ==1000.5496215820312] = 0.0
        warp_image = cv2.imread(base_path +'/warp_image_ray/' + str(i+1) + '.png')
        find_occlusion(rn_opt, rn_base, base_smpl.f)



main()


