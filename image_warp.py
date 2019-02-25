import cv2
import h5py
import argparse
import numpy as np
import cPickle as pkl
from models.smpl import Smpl

from objective import pose_objective, face_objective
import chumpy as ch

from opendr.camera import ProjectPoints
from opendr.renderer import DepthRenderer, ColoredRenderer
from opendr.geometry import Rodrigues
from util import mesh

def take_color_image(rn,base_face_id,opt_face_id, camera, image, f):
    visibility = rn.visibility_image.ravel()
    visible = np.nonzero(visibility != 4294967295)[0]

    consensus_face = np.intersect1d(np.unique(base_face_id), np.unique(opt_face_id))
    diff_face = np.setdiff1d(base_face_id, consensus_face)
    v_id = np.unique(f[base_face_id].ravel())
    diff_v_id = np.unique(f[diff_face].ravel())
    image = image/255.0
    xy = camera.r[v_id].astype(np.int)
    color = image[xy[:, 1],xy[:, 0],:]
    vc = np.zeros_like(camera.v)
    vc[v_id,:] = color

    #vc[diff_v_id,:] = np.zeros(3)
    return vc



# return the visible face projection cordinate in the image
def visible_face_image(rn, camera, f):
    visibility = rn.visibility_image.ravel()
    visible = np.nonzero(visibility != 4294967295)[0]
    face_id = visibility[visible]
    visible_face = f[face_id]
    proj = camera.r
    image_face_coords = proj[visible_face]  # 2D images coordinates of visible faces.
    smpl_v3d = camera.v[visible_face]
    return image_face_coords, visible, face_id, smpl_v3d

def show(rn, rn_opt):
    img_diff = rn.r - rn_opt.r
    cv2.imshow('opt', abs(img_diff))
    cv2.waitKey()

def get_warp_mat(src_point, dst_point):

    src = np.hstack((src_point,np.ones((src_point.shape[0],1))))
    warp_mat = np.linalg.lstsq(src, dst_point)
    return warp_mat[0].T

def get_point_weight(point, tri_points):

    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2


def get_point_weight_Vectorization(point, tri_points):

    tp = tri_points
    # vectors
    v0 = (tp[2,:] - tp[0,:])
    v1 = (tp[1,:] - tp[0,:])
    v2 = (point - tp[0,:])

    dot00 = v0[:,0]**2 + v0[:,1]**2 + v0[:,2]**2
    dot01 = v0[:,0]*v1[:,0] + v0[:,1]*v1[:,1] + v0[:,2]*v1[:,2]
    dot02 = v0[:,0]*v2[:,0] + v0[:,1]*v2[:,1] + v0[:,2]*v2[:,2]
    dot11 = v1[:,0]**2 + v1[:,1]**2 + v1[:,2]**2
    dot12 = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1] + v1[:,2]*v2[:,2]

    div = dot00 * dot11 - dot01 * dot01
    invert = np.zeros_like(div)
    mask = np.where(div!=0)
    invert[mask] = 1.0/div[mask]

    u = (dot11*dot02 - dot01*dot12)*invert
    v = (dot00*dot12 - dot01*dot02)*invert

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2

'''
visible index is the index of pixel projected of smpl
image_face_coords: every visible face's vertices'2D coordinates of smpl model
 
'''


def get_weight(depth_image, visible_index, image_face_coords, smpl_v3d, camera):

    xyz = camera.unproject_depth_image(depth_image)
    tri_face_v3d = xyz[image_face_coords[:, :, 1].astype(np.uint8), image_face_coords[:, :, 0].astype(np.uint8)]

    v3d = xyz.reshape((-1,3))[visible_index]
    print('smpl_v3d',smpl_v3d.shape)
    '''
    height, width = depth_image.shape
    depth = depth_image.flatten()
    print('depth', depth.shape)
    depth = depth[visible_index]
    rows = (np.arange(height*width)/width)[visible_index]
    cols = (np.arange(height*width)%width)[visible_index]
    v3d = np.vstack((cols, rows, depth)).T
    tri_face_3d = np.dstack((image_face_coords[:, :, 0],image_face_coords[:, :, 1], tri_face_depth)).transpose((1, 0, 2))
    '''
    print('tri_face_3d', tri_face_v3d.shape)
    print('v3d', v3d.shape)
    w0, w1, w2 = get_point_weight_Vectorization(v3d, smpl_v3d.transpose((1, 0, 2)))
    return w0, w1, w2


# warp image
def warp_image(src_image, depth_image_opt, visible_index_opt, base_face_id, opt_face_id, image_face_coords_opt,opt_smpl_v3d, base_v, f, opt_camera,track_camera):

    common_face_id = np.intersect1d(np.unique(base_face_id), np.unique(opt_face_id))
    height,width = depth_image_opt.shape
    w0, w1, w2 = get_weight(depth_image_opt, visible_index_opt, image_face_coords_opt, opt_smpl_v3d, opt_camera)
    w0 = w0.reshape((-1,1))
    w1 = w1.reshape((-1,1))
    w2 = w2.reshape((-1,1))
    vertices_of_opt = base_v[f[opt_face_id]]
    v3d_base = vertices_of_opt[:, 0, :]*w0 + vertices_of_opt[:,1,:]*w1 + vertices_of_opt[:,2,:]*w2
    print('v3d_opt', v3d_base.shape)

    # ensure face of every pixel is visible in the opt_smpl
    common_visible_ind = np.where(np.in1d(opt_face_id, common_face_id))[0]
    track_camera.set(v=v3d_base)
    print('common_visible_ind', common_visible_ind.shape)
    pixel_coords = track_camera.r[common_visible_ind]  # 2d coordinates of warp image

    mapx = np.zeros((height*width, ))
    mapy = np.zeros((height * width, ))
    mapx[visible_index_opt[common_visible_ind]]= pixel_coords[:, 0]
    mapy[visible_index_opt[common_visible_ind]] = pixel_coords[:, 1]
    mapy = mapy.reshape((height,  width))
    mapx = mapx.reshape((height, width))
    '''
    dst_image = np.zeros_like(src_image)
    for i in range(mapx.shape[0]):
        for j in range(mapx.shape[1]):
            dst_image[int(mapy[i, j]), int(mapx[i, j]), 0] = src_image[i, j, 0]
            dst_image[int(mapy[i, j]), int(mapx[i, j]), 1] = src_image[i, j, 1]
            dst_image[int(mapy[i, j]), int(mapx[i, j]), 2] = src_image[i, j, 2]
    '''

    src_image = src_image.astype(np.uint8)
    dst_image = cv2.remap(src_image, mapx.astype(np.float32), mapy.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    dst_image = dst_image.astype(np.uint8)
    #cv2.imshow('win', dst_image)
    #cv2.waitKey()
    return dst_image


def face_warp(base_face_id,opt_face_id,base_image_coords,opt_image_coords,f):

    bucker = np.zeros((f.shape[0],2,3))
    common_face, base_index, opt_index = np.intersect1d(base_face_id, opt_face_id, return_indices=True)

    for i in range(common_face.shape[0]):
        wapr_mat = get_warp_mat(opt_image_coords[opt_index[i]],base_image_coords[base_index[i]])
        bucker[common_face[i]] = wapr_mat
    return bucker

def warp_image1(base_visible,opt_visible, base_image_coords, opt_image_coords,src_image,base_face_id,opt_face_id,f):
    bucker = face_warp(base_face_id, opt_face_id, base_image_coords, opt_image_coords, f)
    print(src_image.shape)
    height,width, n_channel = src_image.shape

    src_point = np.zeros((opt_visible.shape[0], 2))
    dst_point = np.zeros((opt_visible.shape[0], 2))
    dst_image = np.zeros_like(src_image)
    src_image = src_image.astype(np.uint8)
    mapx = np.zeros((height*width, ))
    mapy = np.zeros((height * width, ))


    for i in range(opt_visible.shape[0]):

        y = opt_visible[i]%width
        x = opt_visible[i]//width
        warp_mat = bucker[opt_face_id[i]]
        warp_point = warp_mat.dot(np.array([y, x, 1]))
        dst_point[i] = np.array([y, x]).astype(np.int)
        src_point[i] = warp_point.astype(np.int)

    mapx[opt_visible] = src_point[:, 0]
    mapy[opt_visible] = src_point[:, 1]
    mapx = mapx.reshape((height, width))
    mapy = mapy.reshape((height, width))
    dst_image = cv2.remap(src_image,mapx.astype(np.float32),mapy.astype(np.float32),interpolation = cv2.INTER_LINEAR)

    dst_image = dst_image.astype(np.uint8)
    #cv2.imshow('win', dst_image)
    #cv2.waitKey()
    return dst_image

def main():

    base_path = '/home/suoxin/Body/obj1/'
    consensus_file = base_path + 'result2/betas.pkl'
    camera_file = base_path + 'result2/camera.pkl'
    image_file = base_path + 'image/image.hdf5'
    pose_file = base_path + 'result2/reconstructed_poses.hdf5'
    masks_file = base_path + 'mask/masks.hdf5'

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
    #opt_pose = np.average(poses[:, 3:], axis=0)

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

    for i in range(30):

        base_smpl.pose[:] = poses[i]
        base_smpl.trans[:] = trans[i]

        # obtain the model visible face and the every vertice projection image location
        base_image_coords, base_visible, base_face_ind, base_smpl_v3d = visible_face_image(rn_base, camera, base_smpl.f)

        J0 = base_smpl.J[0]
        J0 = J0.reshape((3, 1))
        trans_i = np.array(trans[i].copy())
        pose_i = np.array(poses[i, :3].copy())

        trans_tmp = trans_i.reshape((3, 1))
        trans_tmp = cv2.Rodrigues(pose_i[:3])[0].dot(J0) - J0 - trans_tmp
        trans_tmp = - trans_tmp.reshape((1, 3))

        opt_trans_tmp[:] = trans_tmp.r
        opt_camera_pose_tmp[:] = pose_i

        opt_image_coords, opt_visible, opt_face_ind, opt_smpl_v3d = visible_face_image(rn_opt, opt_camera, opt_smpl.f)
        depth_image_opt = drn_opt.r.copy()
        depth_image_opt[depth_image_opt ==1000.5496215820312] = 0.0
        src_image = cv2.imread(base_path + '/image/' + str(i) + '.png')
        #v3d = camera.unproject_depth_image(depth_image)
        #np.savetxt('/home/suoxin/Body/obj1/result1/point3d.txt', v3d.reshape((-1,3)), fmt='%0.8f')
        #w0, w1, w2=get_weight(depth_image, base_visible, base_image_coords,base_smpl_v3d, camera)
        #dst_image = warp_image(images[i], depth_image, base_visible, base_face_ind, opt_face_ind, base_image_coords,base_smpl_v3d, opt_V,opt_smpl.f, camera,track_camera)
        dst_image = warp_image(src_image, depth_image_opt, opt_visible, base_face_ind, opt_face_ind, opt_image_coords,
                   opt_smpl_v3d, base_smpl.v, base_smpl.f, opt_camera, track_camera)

        #dst_image = warp_image1(base_visible,opt_visible, base_image_coords, opt_image_coords,src_image , base_face_ind, opt_face_ind, base_smpl.f)

        #vc = take_color_image(rn_base, base_face_ind, opt_face_ind, camera, src_image, base_smpl.f)
        #rn_opt.set(vc = vc, bgcolor = ch.zeros(3), num_channels = 3)
        #gray_image = cv2.cvtColor(dst_image, cv2.COLOR_RGB2GRAY)
        #_, dst_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
        #print('dst_mask', np.max(dst_mask))
        #dst_image1 = rn_opt.r * (255.0- dst_mask).reshape(dst_image.shape[0], dst_image.shape[1], -1)
        #dst_image2 = dst_image + dst_image1
        #cv2.imshow('win', dst_image)
        #cv2.waitKey()

        cv2.imwrite(base_path + '/warp_image_ray_opt_pose/' + str(i) + '.png', dst_image)
        #mesh.write('/home/suoxin/Body/obj1/result1/base_smpl0.obj', base_smpl.r, base_smpl.f, vt=vt, ft=ft)



main()

