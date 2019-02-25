import cv2
import numpy as np
from chumpy.ch import Ch
from vendor.smpl.posemapper import posemap
from lib.geometry import visible_boundary_edge_verts
import chumpy as ch


class Square(Ch):
    dterms = ('x', )
    terms = ()

    def compute_r(self):
        print('square')
        return np.dot(self.x.r , self.x.r)

    def compute_dr_wrt(self, wrt):
        if wrt is self.x:

            return 2 * self.x.r


# find the contour of mask
def silh_from_mask(mask):

    if cv2.__version__[0] == '2':
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    silh = np.zeros_like(mask)

    for con in contours:
        cv2.drawContours(silh, [con], 0, 1, 1)

    points = np.vstack(np.where(silh == 1)[::-1]).astype(np.float32).T

    return points

# mask_normal
def mask_normal(points):
    from sklearn.neighbors import NearestNeighbors
    from scipy.optimize import minimize
    neighbors = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(points)
    distances, indices = neighbors.kneighbors(points)
    error = (points[indices] - points.reshape(-1, 1, 2))[:,1:,:]
    normal = np.random.randn(error.shape[0], 2)
    def normal_func(x, A):

        #x0 = np.array([x[0], np.sqrt(1-x[0]*x[0])])
        return np.sum(A.dot(x)**2)

    for i in range(error.shape[0]):
        A = error[i]
        x0 = normal[i]
        cons = {'type': 'eq', 'fun': lambda x: x[0] ** 2 + x[1] ** 2 - 1}
        res = minimize(normal_func, x0 = x0, args=(A), constraints=cons)
        print(res.x)
        normal[i] = res.x

    return normal



# find the minimum contour
def select_points(points, pose_model, camera, rn_b, rn_m):

    v_ids = visible_boundary_edge_verts(rn_b, rn_m)  #boundary ids

    verts = pose_model.v[v_ids]
    project_points = camera[v_ids]
    dist = ((project_points.reshape(-1, 1, 2) - points.reshape(1, -1, 2)) **2).sum(2)# n m 2

    point_id = np.argmin(dist, axis=1)

    return project_points, points[point_id],verts,v_ids



def pose_objective(opt_pose, pose):

    theta = posemap('lrotmin')(np.concatenate((np.zeros(3), pose)))
    opt_theta = posemap('lrotmin')(ch.concatenate((ch.zeros(3), opt_pose)))
    print('opt_theta', opt_theta.shape)
    delta_theta = (opt_theta - theta)

    return delta_theta


def mask_objective(points, pose_model, camera, rn_b, rn_m):

    project_points, selected_points,verts,v_ids = select_points(points, pose_model, camera, rn_b, rn_m)

    e= project_points - selected_points
    print('show_tree', e.show_tree())
    print('mask_objective', e.r.shape)
    return e

'''
concensus triangle quan deng  plus extra triangle
'''
def face_objective(opt_face_length, opt_face_id, face_length, face_id):
    _, opt_ind, ind = np.intersect1d(opt_face_id, face_id, return_indices=True)
    rest_ind = list(set(range(len(face_id))) - set(ind))

    opt_concensuns_length = opt_face_length[opt_ind]
    concensuns_length = face_length[ind]
    rest_length = face_length[rest_ind]

    e = ch.sum((opt_concensuns_length - concensuns_length), axis = 1)
    rest_e = ch.sum(rest_length, axis=1)

    total_e = ch.concatenate((e, rest_e))

    #total_face = (1.0/(np.ones(total_e.shape[0])*total_e.shape[0]))
    print('concensus_face', len(_))
    return total_e #* total_face

def face_area_objective(opt_face_area, opt_face_id, face_area, face_id):
    _, opt_ind, ind = np.intersect1d(opt_face_id, face_id, return_indices=True)
    opt_concensuns_area = ch.abs(opt_face_area[opt_ind])
    concensuns_area = face_area[ind]
    res = 1.0 - ch.sum(opt_concensuns_area)/ch.sum(ch.abs(face_area))
    #res = ch.abs(opt_concensuns_area - concensuns_area)
    print('concensus area', ch.sum(opt_concensuns_area).r, ch.sum(ch.abs(face_area)).r)
    print('result', res.r)
    return res


def depth_objective(base_camera, opt_camera, base_face_ind, opt_face_ind, f):
    face_ind = np.concatenate((base_face_ind, opt_face_ind))
    v_ind = np.unique(f[face_ind].flatten()).astype(np.int)
    v_ind = np.unique(f[base_face_ind].flatten()).astype(np.int)
    return ch.sum(ch.abs(base_camera.v[v_ind, 2] - opt_camera.v[v_ind, 2]))


if __name__ == '__main__':
    image_path = '/home/suoxin/Body/data/chenxin/mask30/000.png'
    mask = cv2.imread(image_path)[:,:,0].astype(np.uint8)

    points = silh_from_mask(mask)
    from sklearn.neighbors import NearestNeighbors
    from scipy.optimize import minimize
    neighbors = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(points)
    distances, indices = neighbors.kneighbors(points)
    error = (points[indices] - points.reshape(-1, 1, 2))[:,1:,:]
    normal = np.random.randn(error.shape[0], 2)

    def normal_func(x, A):

        #x0 = np.array([x[0], np.sqrt(1-x[0]*x[0])])
        return np.sum(A.dot(x)**2)

    for i in range(error.shape[0]):
        A = error[i]
        x0 = normal[i]
        cons = {'type': 'eq', 'fun': lambda x: x[0] ** 2 + x[1] ** 2 - 1}
        res = minimize(normal_func, x0 = x0, args=(A), constraints=cons)
        print(res.x)
    pass
