import cPickle as pkl
from models.smpl import Smpl
import h5py

import numpy as np
from opendr.camera import ProjectPoints
from opendr.renderer import BoundaryRenderer, ColoredRenderer, DepthRenderer
from opendr.lighting import LambertianPointLight
import chumpy as ch
import cv2
from opendr.geometry import Rodrigues
from util import mesh
from opendr.serialization import load_mesh


def align_save():
    refine_betas = '/home/suoxin/Body/obj1/result2/betas.pkl'
    camera_file = '/home/suoxin/Body/obj/camera.pkl'
    pose_data = '/home/suoxin/Body/obj/result/reconstructed_poses.hdf5'
    masks_file = '/home/suoxin/Body/obj/mask/masks.hdf5'
    model = '/home/suoxin/Body/videoavatars/vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'

    vt = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_vt.npy')
    ft = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_ft.npy')

    with open(model, 'rb') as fp:
        model_data = pkl.load(fp)

    with open(camera_file, 'rb') as fp:
        camera_data = pkl.load(fp)

    with open(refine_betas, 'rb') as fp:
        refine_betas = pkl.load(fp)

    pose_data = h5py.File(pose_data, 'r')
    poses = pose_data['pose'][0:30]
    trans = pose_data['trans'][0:30]
    masks = h5py.File(masks_file, 'r')['masks'][0:30]
    base_smpl = Smpl(model_data)
    print(poses.shape)
    base_smpl.betas[:] = np.array(pose_data['betas'], dtype=np.float32)
    #base_smpl.v_personal[:] = refine_betas['v_personal']
    #base_smpl.betas[:] = refine_betas['betas']

    camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'],
                           f=camera_data['camera_f'], k=camera_data['camera_k'], v=base_smpl)

    print('camera_data[camera_c]',camera_data['camera_c'])
    camera_t = camera_data['camera_t']
    camera_rt = camera_data['camera_rt']
    frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}

    for i in range(367):
        mask = np.array(masks[i], dtype=np.uint8)
        pose = np.array(poses[i], dtype=np.float32)
        trans_i = np.array(trans[i], dtype=np.float32)
        base_smpl.pose[:] = pose
        base_smpl.trans[:] = trans_i
        #base_smpl.pose[:3] = np.zeros(3)
        #base_smpl.trans[:] = np.zeros(3)

        #background_image1 = cv2.imread('/home/suoxin/Body/people_snapshot_public/female-1-casual/image/'+str(i)+'.png')
        background_image = np.zeros((masks[i].shape[0], masks[1].shape[1], 3))
        background_image[:,:,0] = masks[i]
        background_image[:, :, 2] = masks[i]
        background_image[:, :, 1] = masks[i]
        rn = ColoredRenderer(camera=camera, v=base_smpl, f=base_smpl.f, vc = np.ones_like(base_smpl),
                             frustum=frustum,background_image = background_image)
        vc = base_smpl.v * 0 + ch.array([1.0, 1.0, 1.0])
        rn.vc = LambertianPointLight(f=base_smpl.f, v=base_smpl, num_verts=len(base_smpl), light_pos=ch.array([0, 0, -1000]), vc=vc,
                                     light_color=ch.array([1., 1., 1.]))
        drn = DepthRenderer(camera=camera, v=base_smpl, f=base_smpl.f, frustum=frustum)
        depth_image = drn.r.copy()
        #print(np.max(depth_image))
        depth_image[depth_image ==1000.5496215820312] = 0.0

        #depth_image = depth_image/np.max(depth_image)
        #v3d = camera.unproject_depth_image(depth_image)
        #np.savetxt('/home/suoxin/Body/obj1/result1/point3d.txt', v3d.reshape((-1,3)), fmt='%0.8f')
        #mesh.write('/home/suoxin/Body/obj/smpl/'+str(i)+'.obj', base_smpl.r, base_smpl.f, vt=vt, ft=ft)
        import scipy.io as sio
        #sio.savemat('/home/suoxin/Body/obj1/result1/depth.mat', {'depth':depth_image})
        #io.imsave('/home/suoxin/Body/obj1/smpl_depth_image/'+str(i)+'.png', (depth_image*255).astype(np.uint8))
        cv2.imshow('win', rn.r)
        cv2.waitKey()


def test_opt_pose():
    consensus_file = '/home/suoxin/Body/obj1/result1/betas.pkl'
    opt_pose_file = '/home/suoxin/Body/obj1/result1/opt_smpl.pkl'
    camera_file = '/home/suoxin/Body/obj1/camera.pkl'
    image_file = '/home/suoxin/Body/obj1/texture_image/image.hdf5'
    pose_file = '/home/suoxin/Body/obj1/result1/reconstructed_poses.hdf5'
    masks_file = '/home/suoxin/Body/obj1/mask/masks.hdf5'

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
    images = h5py.File(image_file, 'r')['masks']

    vt = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_vt.npy')
    ft = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_ft.npy')

    with open(opt_pose_file) as fp:
        opt_pose_data = pkl.load(fp)
    opt_pose = opt_pose_data['opt_pose']
    opt_camera_pose = opt_pose_data['opt_camera_pose']
    opt_trans = opt_pose_data['opt_trans']


    base_smpl = Smpl(model_data)
    base_smpl.betas[:] = consensus_data['betas']
    base_smpl.v_personal[:] = consensus_data['v_personal']

    camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'],
                           f=camera_data['camera_f'], k=camera_data['camera_k'], v=base_smpl)

    frustum = {'near': 0.1, 'far': 1000., 'width': int(camera_data['width']), 'height': int(camera_data['height'])}

    rn_base = ColoredRenderer(camera=camera, frustum=frustum, f=base_smpl.f, vc=np.zeros_like(base_smpl), bgcolor=1, num_channels=1)

    opt_smpl = Smpl(model_data)
    opt_smpl.betas[:] = consensus_data['betas']
    opt_smpl.v_personal[:] = consensus_data['v_personal']

    opt_trans_tmp = ch.zeros(3)
    opt_camera_pose_tmp = ch.zeros(3)

    opt_smpl.pose = ch.concatenate((ch.zeros(3), opt_pose))
    opt_V = opt_trans_tmp.reshape((1, 3)) + (Rodrigues(opt_camera_pose_tmp).dot(opt_smpl.T)).T

    opt_camera = ProjectPoints(t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'], f=camera_data['camera_f'], k=camera_data['camera_k'], v=opt_V)
    rn_opt = ColoredRenderer(camera=opt_camera, frustum=frustum, f=opt_smpl.f, vc=np.zeros_like(opt_smpl), bgcolor=1, num_channels=1)

    for i in range(30):

        opt_trans_tmp = opt_trans[i]
        opt_camera_pose_tmp = opt_camera_pose[i]

        visibility = rn_opt.visibility_image.ravel()
        visible = np.nonzero(visibility != 4294967295)[0]
        face = np.unique(visibility[visible])
        print('visble face number', face.shape)

        cv2.imshow('win', rn_opt.r)
        cv2.waitKey()


align_save()


