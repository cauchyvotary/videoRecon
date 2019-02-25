import cv2
import h5py
import argparse
import numpy as np
import chumpy as ch
import cPickle as pkl

from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
from opendr.filters import gaussian_pyramid

from util import im
from util.logger import log
from lib.frame import FrameData
from models.smpl import Smpl, copy_smpl, joints_coco
from models.bodyparts import faces_no_hands

from vendor.smplify.sphere_collisions import SphereCollisions
from vendor.smplify.robustifiers import GMOf

model = '/home/suoxin/Body/videoavatars/vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
# load smpl model
with open(model, 'rb') as fp:
    model_data = pkl.load(fp)



smpl = Smpl(model_data)
file = h5py.File('/home/suoxin/Body/obj1/result/reconstructed_poses.hdf5', 'r')
pose = file['pose']
trans = file['trans']
betas = file['betas']

smpl.trans = trans[2, :]
smpl.pose =pose[2, :]
smpl.betas = betas
from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight

rn = ColoredRenderer()

# Assign attributes to renderer
w, h = (640, 480)

rn.camera = ProjectPoints(v=smpl, rt=np.zeros(3), t=np.array([0, 0, 3.]), f=np.array([w, w]),
                          c=np.array([w, h]) / 2., k=np.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=smpl, f=smpl.f, bgcolor=np.zeros(3))

# Construct point light source
rn.vc = LambertianPointLight(
    f=smpl.f,
    v=rn.v,
    num_verts=len(smpl),
    light_pos=np.array([-1000, -1000, -2000]),
    vc=np.ones_like(smpl) * .9,
    light_color=np.array([1., 1., 1.]))

# Show it using OpenCV
import cv2

cv2.imshow('render_SMPL', rn.r)
print ('..Print any key while on the display window')
cv2.waitKey(0)
cv2.destroyAllWindows()

