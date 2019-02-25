import skimage.io as io
import h5py
import numpy as np
import cv2
import pickle
from models.smpl import Smpl, copy_smpl, joints_coco
from util import im, mesh


def plot_joints():
    pose = np.load('/home/suoxin/Body/obj/joints/9.png_pose.npz')['pose'].T
    print(pose[:,:3])
    image = io.imread('/home/suoxin/Body/obj/image/9.png')
    for i in pose[:,:2]:
        cv2.circle(image, tuple(i.astype(np.int)), 3, (0, 0, 0.8), -1)
        cv2.imshow('ss', image)
        cv2.waitKey()

def plot_pose(path,model_file,obj_out):
    pose_data = h5py.File(path, 'r')
    poses = pose_data['pose'][0:30]
    trans = pose_data['trans'][0:30]
    shape = pose_data['betas']
    with open(model_file, 'rb') as fp:
        model_data = pickle.load(fp)
    with open('/home/suoxin/Body/obj1/betas.pkl', 'rb') as fp:
        betas = pickle.load(fp)
    v_personal = betas['v_personal']

    beta = betas['betas']

    base_smpl = Smpl(model_data)
    base_smpl.betas[:] = np.array(beta, dtype=np.float32)
    base_smpl.v_personal[:] = np.array(v_personal,dtype =np.float32)
    print(v_personal)
    for i in range(30):
        base_smpl.pose[:] = np.array(pose_data['pose'][i])
        base_smpl.trans[:] = np.array(pose_data['trans'][i])
        vt = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_vt.npy')
        ft = np.load('/home/suoxin/Body/videoavatars/assets/basicModel_ft.npy')
        mesh.write(obj_out+str(i)+'.obj', base_smpl.r, base_smpl.f, vt=vt, ft=ft)



model_file = '/home/suoxin/Body/videoavatars/vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
obj_out = '/home/suoxin/Body/result/'
plot_joints()

