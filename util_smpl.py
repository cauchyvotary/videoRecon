import numpy as np
from opendr.renderer import ColoredRenderer
import chumpy as ch
import cv2



def pose_trans(n, poses,trans):

    new_poses_total = []
    new_trans_total = []
    for i in range(n):
        new_pose = poses[i]
        new_trans = trans[i]
        new_poses_total.append(new_pose)
        new_trans_total.append(new_trans)

    return new_poses_total, new_trans_total


'''
ensure which face visible by render
'''

def visible_check(smpl, rn_vis, camera):

    visibility = rn_vis.visibility_image.ravel()
    visible = np.nonzero(visibility != 4294967295)[0]
    face = np.unique(visibility[visible])
    f = smpl.f
    select_face = f[face]
    vert = camera[select_face]
    e1 = vert[:, 0, :] - vert[:, 1, :]
    e2 = vert[:, 1, :] - vert[:, 2, :]
    e3 = vert[:, 2, :] - vert[:, 0, :]
    l1 = np.sqrt(ch.sum(e1 ** 2, axis=1)).reshape(-1, 1)
    l2 = ch.sqrt(ch.sum(e2 ** 2, axis=1)).reshape(-1, 1)
    l3 = ch.sqrt(ch.sum(e3 ** 2, axis=1)).reshape(-1, 1)

    face_len = ch.hstack((l1, l2, l3))
    print('face_len', face_len.shape)
    return face_len, face


def visible_area(smpl, rn_vis, camera):
    visibility = rn_vis.visibility_image.ravel()
    visible = np.nonzero(visibility != 4294967295)[0]
    face = np.unique(visibility[visible])
    f = smpl.f
    select_face = f[face]
    vert = camera[select_face]
    e1 = vert[:, 1, :] - vert[:, 0, :]
    e2 = vert[:, 2, :] - vert[:, 0, :]

    tri_area = ch.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    print('tri_area', tri_area.shape)
    #area = ch.sum(tri_area)
    return tri_area, face
