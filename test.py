import cv2
import h5py
import argparse
import numpy as np
import cPickle as pkl
from models.smpl import Smpl
from util.logger import log
from objective import pose_objective, face_objective
import chumpy as ch
from opendr.geometry import TriNormals
from opendr.camera import ProjectPoints
from opendr.renderer import DepthRenderer, ColoredRenderer
from opendr.geometry import Rodrigues
from util import mesh

def take_color_image(rn,rn_opt, camera,opt_camera, image, f):
    visibility = rn.visibility_image.ravel()
    visible = np.nonzero(visibility != 4294967295)[0]

    visibility_opt = rn_opt.visibility_image.ravel()
    visible_opt = np.nonzero(visibility_opt != 4294967295)[0]
    consensus_face = np.intersect1d(np.unique(visibility[visible]), np.unique(visibility_opt[visible_opt]))
    face_id = np.unique(visibility[visible])
    diff_face = np.setdiff1d(face_id, consensus_face)
    print('consensus_face', np.unique(visibility[visible]).shape)
    print('diff_face', diff_face.shape)
    v_id = np.unique(f[face_id].ravel())
    diff_v_id = np.unique(f[diff_face].ravel())
    image = image/255.0
    xy = camera.r[v_id].astype(np.int)
    color = image[xy[:, 1],xy[:, 0],:]

    xy_opt = opt_camera.r[f[diff_face]]
    xy_opt = np.floor((xy_opt[:,0,:] + xy_opt[:,1,:] + xy_opt[:,2,:])/3.0)
    visibility_opt = visibility_opt.reshape(rn.shape)
    #debug_f0 = visibility_opt[156,575]
    #debug_f1 = visibility_opt[173,590]
    #print('debug_f0', debug_f0)
    #print('debug_f1', debug_f1)
    #print('debug_f3', visibility_opt[173,560])
    #print('debug_f4', visibility_opt[165, 575])
    occlu_face = visibility_opt[xy_opt[:,1].astype(np.int), xy_opt[:,0].astype(np.int)]


    vc = np.zeros_like(camera.v)
    v = opt_camera.v
    normals = TriNormals(v, f).reshape(-1,3)
    n = normals[np.unique(diff_face)].r
    #import scipy.io as sio
    #sio.savemat('/home/suoxin/Body/people_snapshot_public/male-2-sport/blank_image/normal.mat',{'n': n})
    #sio.savemat('/home/suoxin/Body/people_snapshot_public/male-2-sport/blank_image/diff_face.mat', {'diff_face': diff_face})
    #sio.savemat('/home/suoxin/Body/people_snapshot_public/male-2-sport/blank_image/occlu_face.mat', {'occlu_face': occlu_face})

    vc[v_id,:] = color
    vc[v_id,:] = np.ones(3)
    vc[diff_v_id,:] = np.zeros(3)
    return vc


def objective(image,mask):
    height, width = image.shape
    #image = image * mask
    #image = image + 1.0-mask
    rows = (np.arange(height * width) / width).reshape(height, width).astype(np.int)
    cols = (np.arange(height * width) % width).reshape(height, width).astype(np.int)
    cols_origin = cols.copy()
    cols_origin[:, 0:width-2] = cols_origin[:, 0:width-2] + 1

    rows_origin = rows.copy()
    rows_origin[0:height-2, :] = rows_origin[0:height-2, :] + 1

    image_diff_row = (image[rows_origin, cols] - image[rows, cols])
    image_diff_col = (image[rows, cols_origin] - image[rows, cols])
    cv2.imshow('diff', image_diff_col.r)
    cv2.waitKey()
    return image_diff_col


def main():

    consensus_file = '/home/suoxin/Body/people_snapshot_public/male-2-sport/consensus.pkl'

    camera_file = '/home/suoxin/Body/people_snapshot_public/male-2-sport/camera.pkl'
    image_file = '/home/suoxin/Body/people_snapshot_public/male-2-sport/image.hdf5'
    pose_file = '/home/suoxin/Body/people_snapshot_public/male-2-sport/reconstructed_poses.hdf5'
    masks_file = '/home/suoxin/Body/people_snapshot_public/male-2-sport/masks.hdf5'

    model = '/home/suoxin/Body/videoavatars/vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    #opt_pose_file = '/home/suoxin/Body/data/chenxin/opt_smpl1.pkl'


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
    #images = h5py.File(image_file, 'r')['masks']

    vt = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_vt.npy')
    ft = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_ft.npy')
    '''
    with open(opt_pose_file) as fp:
        opt_pose_data = pkl.load(fp)
    opt_pose = opt_pose_data['opt_pose']
    opt_camera_pose = opt_pose_data['opt_camera_pose']
    opt_trans = opt_pose_data['opt_trans']
    '''

    opt_pose = poses[1][3:]

    opt_camera_pose = np.zeros((poses.shape[0],3))

    base_smpl = Smpl(model_data)
    base_smpl.betas[:] = consensus_data['betas']
    base_smpl.v_personal[:] = consensus_data['v_personal']

    camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'],
                           f=camera_data['camera_f'], k=camera_data['camera_k'], v=base_smpl)

    frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}

    rn_base = ColoredRenderer(camera=camera, frustum=frustum, f=base_smpl.f, vc=np.zeros_like(base_smpl), bgcolor=np.zeros(3), num_channels=3)
    drn = DepthRenderer(camera=camera, v=base_smpl, f=base_smpl.f, frustum=frustum)

    opt_smpl = Smpl(model_data)
    opt_smpl.betas[:] = consensus_data['betas']
    opt_smpl.v_personal[:] = consensus_data['v_personal']

    opt_trans_tmp = ch.zeros(3)
    opt_camera_pose_tmp = ch.zeros(3)

    opt_smpl.pose = ch.concatenate((ch.zeros(3), opt_pose))
    opt_V = opt_trans_tmp.reshape((1, 3)) + (Rodrigues(opt_camera_pose_tmp).dot(opt_smpl.T)).T

    opt_camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'], f=camera_data['camera_f'], k=camera_data['camera_k'], v=opt_V)
    rn_opt = ColoredRenderer(camera=opt_camera, frustum=frustum, f=opt_smpl.f, vc=np.zeros_like(opt_smpl), bgcolor=ch.zeros(3), num_channels=3)
    drn_opt = DepthRenderer(camera=opt_camera, f=base_smpl.f, frustum=frustum)

    track_camera = ProjectPoints(t=np.zeros(3),rt=np.zeros(3), c=camera_data['camera_c'], f = camera_data['camera_f'], k=camera_data['camera_k'])

    for i in range(0, 1):

        base_smpl.pose[:] = poses[i]
        base_smpl.trans[:] = trans[i]
        import skimage.io as io
        image = cv2.imread('/home/suoxin/Body/people_snapshot_public/male-2-sport/image/' + str(i) + '.png')

        #rn_base.set(vc = vc)
        #cv2.imshow('win', rn_base.r)
        #cv2.waitKey()

        J0 = base_smpl.J[0]
        J0 = J0.reshape((3, 1))
        trans_i = np.array(trans[i].copy())
        pose_i = np.array(poses[i, :3].copy())

        trans_tmp = trans_i.reshape((3, 1))
        trans_tmp = cv2.Rodrigues(pose_i[:3])[0].dot(J0) - J0 - trans_tmp
        trans_tmp = - trans_tmp.reshape((1, 3))

        opt_trans_tmp[:] = trans_tmp.r
        opt_camera_pose_tmp[:] = pose_i
        rn_opt.set(bgcolor=ch.zeros(1), num_channels=1)
        rn_base.set(bgcolor=ch.zeros(1), num_channels=1)
        vc = take_color_image(rn_base, rn_opt, camera,opt_camera, image, base_smpl.f)

        rn_opt.set(vc = vc)
        rn_opt.set(bgcolor=ch.zeros(1), num_channels=1)
        cv2.imshow('before', rn_opt.r)
        cv2.waitKey()
        #cv2.imwrite('/home/suoxin/Body/people_snapshot_public/male-2-sport/render_image/'+str(i)+'.png', (rn_opt.r*255).astype(np.int))
        #cv2.imwrite('/home/suoxin/Body/people_snapshot_public/male-2-sport/blank_image/0.png', (rn_opt.r*255).astype(np.int))
        '''
        visibility = rn_opt.visibility_image.ravel()
        visible = np.nonzero(visibility == 4294967295)[0]
        visibility[visible] = 0
        face = visibility.reshape(rn_opt.shape) * [rn_opt.r<0.5]
        import scipy.io as sio
        sio.savemat('/home/suoxin/Body/people_snapshot_public/male-2-sport/blank_image/rn_opt_face.mat',{'face1':np.unique(face)})
        cv2.imshow('win', rn_opt.r)
        sio.savemat('/home/suoxin/Body/people_snapshot_public/male-2-sport/blank_image/image.mat',
                    {'image':rn_opt.r})
        cv2.waitKey()
        '''

        E = {}
        mask = rn_opt.r
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        E_raw = objective(rn_opt,mask)
        from opendr.filters import gaussian_pyramid
        E_pyr = gaussian_pyramid(E_raw, n_levels=6, normalization='size')
        #print('E[image_diff]', E['image_diff'])
        log.info('## Run optimization')
        '''
        ch.minimize(
            E_pyr,
            [opt_camera_pose_tmp, opt_trans_tmp],
            method='dogleg',
            options={'maxiter': 15, 'e_3': 0.001}
        )
        
        vc = take_color_image(rn_base, rn_opt, camera,opt_camera, image, base_smpl.f)
        rn_opt.set(vc = vc)
        cv2.imwrite('/home/suoxin/Body/people_snapshot_public/male-2-sport/blank_image/1.png', (rn_opt.r*255).astype(np.int))
        cv2.imshow('after gaussian', rn_opt.r)
        cv2.waitKey()
        '''


        ch.minimize(
            E_raw,
            [opt_camera_pose_tmp, opt_trans_tmp]
        )
        vc = take_color_image(rn_base, rn_opt, camera, opt_camera, image, base_smpl.f)
        rn_opt.set(vc = vc)
        cv2.imwrite('/home/suoxin/Body/people_snapshot_public/male-2-sport/blank_image/2.png', (rn_opt.r*255).astype(np.int))
        cv2.imshow('after', rn_opt.r)
        cv2.waitKey()
        import skimage.io as io
        #src_image = io.imread('/home/suoxin/Body/people_snapshot_public/male-2-sport/image/0.png')
        



main()
