#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import h5py
import argparse
import numpy as np
import chumpy as ch
import cPickle as pkl

from models.smpl import Smpl, copy_smpl

from opendr.camera import ProjectPoints
from opendr.renderer import DepthRenderer, ColoredRenderer

from skimage.transform import pyramid_gaussian

from flow_loss_funcs import CorrFlowDataObj, WarpFlowDataObj, Try
from vis_tool import draw_correspondence


def init_corr_warp_flow(base_model, poses, trans, optimal_model, optimal_rt, optimal_t, image_data, mask_data,
                        camera_data, level=3, write_folder=None):
    frame_num = image_data.shape[0]

    color_pyra = []
    mask_pyra = []
    for frame_idx in range(frame_num):
        img = image_data[frame_idx].astype(np.uint8)
        mask = mask_data[frame_idx].astype(np.uint8) * 255
        color_pyra.append(tuple(pyramid_gaussian(image=img, max_layer=level)))
        mask_pyra.append(tuple(pyramid_gaussian(image=mask, max_layer=level)))

    all_level_keep = []
    for l in range(level + 1):
        all_keep = {'uv_all': [], 'uv_valid_mask': [], 'faceid4points': [], 'bary4points': [], 'warp_flow': [],
                    'neighbour_corr_flow': [], 'uv_all_reg_edge': [], 'color': [], 'mask': [], 'depth_img_keep': []}

        height, width = color_pyra[0][l].shape[0:2]
        alpha = 1.0 / np.power(2.0, l)
        K = np.array([[camera_data['camera_f'][0] * alpha, 0, camera_data['camera_c'][0] * alpha],
                      [0, camera_data['camera_f'][1] * alpha, camera_data['camera_c'][1] * alpha],
                      [0, 0, 1]], dtype=np.float)
        all_keep['height'] = height
        all_keep['width'] = width
        all_keep['K'] = K
        frustum = {'near': 0.1, 'far': 10.,
                   'width': int(width), 'height': int(height)}

        original_cam = ProjectPoints(v=base_model.v, t=np.zeros(3), rt=np.zeros(3), c=np.array([K[0, 2], K[1, 2]]),
                                     f=np.array([K[0, 0], K[1, 1]]), k=camera_data['camera_k'])

        d_render = DepthRenderer(f=base_model.f, frustum=frustum, camera=original_cam, v=base_model.v)

        # for each image and mask
        # for frame_idx in range(0, frame_num):
        for frame_idx in range(0, frame_num):
            base_model.pose[:] = poses[frame_idx]
            base_model.trans[:] = trans[frame_idx]
            mask = mask_pyra[frame_idx][l] > 0.5

            all_keep['color'].append(color_pyra[frame_idx][l].astype(np.float32))
            all_keep['mask'].append(mask)

            # TODO : should mark this d_img, sometimes chumpy seems not update
            d_img = d_render.r
            d_vis = d_render.visibility_image
            # mask = np.logical_and(original_mask, d_img < frustum['far'] - 1e-4, d_vis != 4294967295)

            all_keep['depth_img_keep'].append(d_img)
            cv2.imshow('debug', mask.astype(np.uint8) * 255)
            cv2.waitKey(1000)
            cv2.imshow('debug', d_img.astype(np.float) / np.max(d_img))
            cv2.waitKey(1000)

            # select d point
            vu_all = np.nonzero(mask)
            uv_all = np.array(zip(vu_all[1], vu_all[0]))  # all uv needed to consider
            uv_valid_mask = np.logical_and(d_img[mask] < frustum['far'] - 1e-4, d_vis[mask] != 4294967295)
            uv = uv_all[uv_valid_mask]  # valid uv that can be analyzed with template model

            uvd = np.hstack((uv, d_img[mask][uv_valid_mask].reshape((-1, 1))))
            lifted_points = original_cam.unproject_points(uvd)

            points_face_idx = d_vis[mask][uv_valid_mask]

            # calculate barycentric coordinate
            faces4points = base_model.f[points_face_idx]
            v0 = base_model.r[faces4points[:, 0].ravel()]
            v1 = base_model.r[faces4points[:, 1].ravel()]
            v2 = base_model.r[faces4points[:, 2].ravel()]

            vn = np.cross(v1 - v0, v2 - v0)
            vn = vn / np.sqrt(np.sum(vn ** 2, axis=1)).reshape((-1, 1))

            lifted_confidence = np.sum((lifted_points - v0) * vn, axis=1)

            LIFTED_CONFIDENCE_TH = 1e-3
            bary4points = np.zeros((len(lifted_confidence), 3))
            for p_id in range(0, len(lifted_confidence)):
                if np.abs(lifted_confidence[p_id]) < LIFTED_CONFIDENCE_TH:
                    m = np.vstack((v1[p_id] - v0[p_id], v2[p_id] - v0[p_id]))
                    b, c = np.linalg.solve(m.dot(m.T), m.dot((lifted_points[p_id] - v0[p_id]).reshape((-1, 1))))
                    a = 1 - b - c

                    bary4points[p_id, :] = [a, b, c]

            # update valid mask
            confidence_mask = np.abs(lifted_confidence) < LIFTED_CONFIDENCE_TH
            uv_valid_mask[uv_valid_mask] = confidence_mask
            uv = uv[confidence_mask]
            points_face_idx = points_face_idx[confidence_mask]
            faces4points = faces4points[confidence_mask]
            bary4points = bary4points[confidence_mask]

            # set optimal points
            v0 = optimal_model.r[faces4points[:, 0].ravel()]
            v1 = optimal_model.r[faces4points[:, 1].ravel()]
            v2 = optimal_model.r[faces4points[:, 2].ravel()]

            # set optimal cam
            optimal_points = v0 * bary4points[:, 0].reshape(-1, 1) + v1 * bary4points[:, 1].reshape(-1, 1) + \
                             v2 * bary4points[:, 2].reshape(-1, 1)
            optimal_cam = ProjectPoints(v=optimal_points, t=optimal_t[frame_idx],  # t=base_model.trans.r[:] + base_model.J.r[0],
                                        rt=optimal_rt[frame_idx],                    # rt=base_model.pose.r[0:3],
                                        c=np.array([K[0, 2], K[1, 2]]),
                                        f=np.array([K[0, 0], K[1, 1]]), k=camera_data['camera_k'])

            warped_uv = optimal_cam.r
            warp_flow = warped_uv - uv

            # # debug
            # color_img = image_data[frame_idx].astype(np.uint8)
            # warp_img = np.zeros_like(color_img, dtype=color_img.dtype)
            # warp_img[np.round(warped_uv[:, 1]).astype(np.int), np.round(warped_uv[:, 0]).astype(np.int)] = color_img[uv[:, 1], uv[:, 0]]
            # cv2.imshow('debug', warp_img)
            # cv2.waitKey()
            # #### debug done

            all_keep['uv_all'].append(uv_all)
            all_keep['uv_valid_mask'].append(uv_valid_mask)
            all_keep['faceid4points'].append(points_face_idx)
            all_keep['bary4points'].append(bary4points)
            all_keep['warp_flow'].append(warp_flow)

            # get correspondences from adjacent views
            NEIGHBOUR_RADIUS = 1
            neighbour_frame_idx = map(lambda x: x % frame_num, range(frame_idx - NEIGHBOUR_RADIUS, frame_idx)) + \
                                  map(lambda x: x % frame_num, range(frame_idx + 1, frame_idx + NEIGHBOUR_RADIUS + 1))
            corr_flow = {}
            for neigh_fid in neighbour_frame_idx:
                base_model.pose[:] = poses[neigh_fid]
                base_model.trans[:] = trans[neigh_fid]

                v0 = base_model.r[faces4points[:, 0].ravel()]
                v1 = base_model.r[faces4points[:, 1].ravel()]
                v2 = base_model.r[faces4points[:, 2].ravel()]

                neigh_points = v0 * bary4points[:, 0].reshape(-1, 1) + v1 * bary4points[:, 1].reshape(-1, 1) + \
                               v2 * bary4points[:, 2].reshape(-1, 1)

                nei_cam = ProjectPoints(v=neigh_points, t=np.zeros(3), rt=np.zeros(3), c=camera_data['camera_c'],
                                        f=camera_data['camera_f'], k=camera_data['camera_k'])
                nei_uv = nei_cam.r
                corr_flow[neigh_fid] = nei_uv - uv

            all_keep['neighbour_corr_flow'].append(corr_flow)

            # calculate the reg relation for current frame
            tmp_mask = -np.ones_like(mask, dtype=np.int)
            tmp_mask[uv_all[:, 1], uv_all[:, 0]] = range(uv_all.shape[0])
            ud_I1 = np.array(range(mask.shape[0] - 1)).repeat(mask.shape[1])
            ud_I2 = np.array(range(1, mask.shape[0])).repeat(mask.shape[1])
            ud_J = np.array(range(mask.shape[1])).reshape((1, -1)).repeat(mask.shape[0] - 1, axis=0).ravel()
            lr_I = np.array(range(mask.shape[0])).repeat(mask.shape[1] - 1)
            lr_J1 = np.array(range(mask.shape[1] - 1)).reshape((1, -1)).repeat(mask.shape[0], axis=0).ravel()
            lr_J2 = np.array(range(1, mask.shape[1])).reshape((1, -1)).repeat(mask.shape[0], axis=0).ravel()
            edges = np.vstack(
                (np.hstack((tmp_mask[ud_I1, ud_J].reshape((-1, 1)), tmp_mask[ud_I2, ud_J].reshape((-1, 1)))),
                 np.hstack((tmp_mask[lr_I, lr_J1].reshape((-1, 1)), tmp_mask[lr_I, lr_J2].reshape((-1, 1))))))
            edges_valid = np.logical_and(edges[:, 0] != -1, edges[:, 1] != -1)
            reg_idx_uvall = edges[edges_valid]

            all_keep['uv_all_reg_edge'].append(reg_idx_uvall)

        all_level_keep.append(all_keep)

    if write_folder is not None:
        for i in range(level + 1):
            f_name = '/keep_data_level_{:02d}.pkl'.format(i)
            with open(write_folder + f_name, 'wb') as f:
                pkl.dump(all_level_keep[i], f, pkl.HIGHEST_PROTOCOL)


def calculate_essential(r2, t2, r1, t1):
    R1 = cv2.Rodrigues(r1)[0]
    R2 = cv2.Rodrigues(r2)[0]
    R = R2.dot(R1.T)
    t = -R.dot(t1) + t2
    t_tilde = np.zeros((3, 3))
    t_tilde[0, 1] = -t[2]
    t_tilde[0, 2] = t[1]
    t_tilde[1, 0] = t[2]
    t_tilde[1, 2] = -t[0]
    t_tilde[2, 0] = -t[1]
    t_tilde[2, 1] = t[0]
    E_ts = t_tilde.dot(R)

    return E_ts


def main(pose_file, shape_file, image_file, mask_file, camera_file, model_file):
    pose_data = h5py.File(pose_file, 'r')
    poses = pose_data['pose']
    trans = pose_data['trans']

    with open(shape_file, 'rb') as f:
        shape_data = pkl.load(f)
    beta = shape_data['betas']
    v_personal = shape_data['v_personal']

    image_data = h5py.File(image_file, 'r')['color']
    mask_data = h5py.File(mask_file, 'r')['masks']

    # ## debug
    # pyra = tuple(pyramid_gaussian(image=mask_data[0].astype(np.uint8) * 255, max_layer = 3))
    # cv2.imshow('debug', pyra[0])
    # cv2.waitKey()
    # cv2.imshow('debug', (pyra[0] > 0.5).astype(np.uint8) * 255)
    # cv2.waitKey()
    # cv2.imshow('debug', pyra[1])
    # cv2.waitKey()
    # cv2.imshow('debug', (pyra[1] > 0.5).astype(np.uint8) * 255)
    # cv2.waitKey()
    # cv2.imshow('debug', pyra[2])
    # cv2.waitKey()
    # cv2.imshow('debug', (pyra[2] > 0.5).astype(np.uint8) * 255)
    # cv2.waitKey()
    # cv2.imshow('debug', pyra[3])
    # cv2.waitKey()
    # cv2.imshow('debug', (pyra[3] > 0.5).astype(np.uint8) * 255)
    # cv2.waitKey()
    # #### debug done

    with open(camera_file, 'rb') as f:
        camera_data = pkl.load(f)

    with open(model_file, 'rb') as f:
        model_data = pkl.load(f)

    # model for each frame
    base_model = Smpl(model=model_data, betas=beta, v_personal=v_personal)

    # model of optimal pose
    optimal_model = Smpl(model=model_data, betas=beta, v_personal=v_personal)
    # for now we just FAKE the optimal human pose and camera pose
    optimal_model.pose[:] = poses[15]
    optimal_model.pose[0:3] = 0

    frame_num = image_data.shape[0]

    # TODO: get the real optimal rotation and translation for camera
    optimal_rt = np.zeros((frame_num, 3))
    optimal_t = np.zeros((frame_num, 3))

    for frame_idx in range(frame_num):
        base_model.trans[:] = trans[frame_idx]
        base_model.pose[:] = poses[frame_idx]
        optimal_t[frame_idx] = base_model.J.r[0] + base_model.trans.r[:]
        optimal_rt[frame_idx] = base_model.pose.r[0:3]

    # get initial corr_flow and warp_flow
    init_corr_warp_flow(base_model, poses, trans, optimal_model, optimal_rt, optimal_t, image_data, mask_data,
                        camera_data, level=3, write_folder='../data/')
    # with open('../data/keep_data_level_00.pkl', 'rb') as f:
    #     all_keep = pkl.load(f)

    '''
    optimize the corr flow
    '''
    optimal_corr_flow_keep = []
    weight = {'data_weight': 1.0, 'reg_weight': 1.0}
    for frame_idx in range(frame_num):
        cur_opt_corr_flow = {}
        neigh_index = all_keep['neighbour_corr_flow'][frame_idx].keys()
        uv_all = all_keep['uv_all'][frame_idx]
        uv_valid_mask = all_keep['uv_valid_mask'][frame_idx]
        src_img = all_keep['color'][frame_idx]
        reg_idx = all_keep['uv_all_reg_edge'][frame_idx]

        for nid in neigh_index:
            cf_cur = ch.Ch(np.zeros((uv_all.shape[0], 2)))
            cur_opt_corr_flow[nid] = cf_cur
            neigh_img = all_keep['color'][nid]

            cf_obj = CorrFlowDataObj(src_img=src_img, target_img=neigh_img,
                                     uv_all=uv_all, corr_flow_all=cf_cur)

            data_obj = weight['data_weight'] * (cf_cur[uv_valid_mask] - all_keep['neighbour_corr_flow'][frame_idx][nid])

            reg_obj = weight['reg_weight'] * (cf_cur[reg_idx[:, 0]] - cf_cur[reg_idx[:, 1]])

            ch.minimize({'cf_obj': cf_obj, 'data_obj': data_obj, 'reg_obj': reg_obj},
                        [cf_cur],
                        method='dogleg',
                        options={
                            'e_3': .01,
                        })

            # draw_correspondence(src_img, neigh_img, uv_all[uv_valid_mask],
            #                     uv_all[uv_valid_mask] + np.round(all_keep['neighbour_corr_flow'][frame_idx][nid]).astype(np.int))
            # draw_correspondence(src_img, neigh_img, uv_all, uv_all + np.round(cf_cur.r).astype(np.int))

        optimal_corr_flow_keep.append(cur_opt_corr_flow)
    # write the dict
    with open('../data/optimal_corr_flow_level_00.pkl', 'wb') as f:
        pkl.dump(optimal_corr_flow_keep, f, pkl.HIGHEST_PROTOCOL)
    # with open('../data/optimal_corr_flow_level_00.pkl', 'rb') as f:
    #    optimal_corr_flow_keep = pkl.load(f)

    '''
     warp flow optimization
    '''
    optimal_warp_flow_keep = []
    wf_opt_weight = {'data_weight': 1.0, 'reg_weight': 1.0}
    for frame_idx in range(frame_num):
        uv_all = all_keep['uv_all'][frame_idx]

        wf_cur = ch.Ch(np.zeros((uv_all.shape[0], 2)))
        optimal_warp_flow_keep.append(wf_cur)

        uv_valid_mask = all_keep['uv_valid_mask'][frame_idx]
        reg_idx = all_keep['uv_all_reg_edge'][frame_idx]

        # initialize by data term and reg term
        data_obj = wf_opt_weight['data_weight'] * (wf_cur[uv_valid_mask] - all_keep['warp_flow'][frame_idx])
        reg_obj = wf_opt_weight['reg_weight'] * (wf_cur[reg_idx[:, 0]] - wf_cur[reg_idx[:, 1]])

        ch.minimize({'data_obj': data_obj, 'reg_obj': reg_obj}, [wf_cur],
                    method='dogleg', options={'e_3': .01})

        # debug
        color_img = all_keep['color'][frame_idx]
        warp_img = np.zeros_like(color_img, dtype=color_img.dtype)
        debug_warp_uv = np.round(uv_all + wf_cur.r).astype(np.int)
        debug_warp_uv[:, 0] = np.maximum(np.minimum(all_keep['width'] - 1, debug_warp_uv[:, 0]), 0)
        debug_warp_uv[:, 1] = np.maximum(np.minimum(all_keep['height'] - 1, debug_warp_uv[:, 1]), 0)
        warp_img[debug_warp_uv[:, 1], debug_warp_uv[:, 0]] = color_img[uv_all[:, 1], uv_all[:, 0]]
        cv2.imshow('debug', warp_img)
        cv2.waitKey(1000)
        # debug done

    with open('../data/optimal_warp_flow_init_level_00.pkl', 'wb') as f:
        pkl.dump(optimal_warp_flow_keep, f, pkl.HIGHEST_PROTOCOL)
    #with open('../data/optimal_warp_flow_init_level_00.pkl', 'rb') as f:
    #    optimal_warp_flow_keep = pkl.load(f)

    '''
     continue optimizing with differential rigidity framwork
    '''
    for frame_idx in range(frame_num):
        neigh_index = all_keep['neighbour_corr_flow'][frame_idx].keys()
        uv_all = all_keep['uv_all'][frame_idx]
        uv_valid_mask = all_keep['uv_valid_mask'][frame_idx]
        reg_idx = all_keep['uv_all_reg_edge'][frame_idx]
        wf_cur = optimal_warp_flow_keep[frame_idx]

        for nid in neigh_index:
            target_uv_all = all_keep['uv_all'][nid]
            target_wf_cur = optimal_warp_flow_keep[nid]
            E_ts = calculate_essential(optimal_rt[nid], optimal_t[nid], optimal_rt[frame_idx], optimal_t[frame_idx])
            s2t_cf = optimal_corr_flow_keep[frame_idx][nid]

            wf_obj = WarpFlowDataObj(uv_all=uv_all, K=all_keep['K'], E_ts=E_ts,
                                     target_uv_all=target_uv_all, height=all_keep['height'],
                                     width=all_keep['width'],
                                     warp_flow_all=wf_cur,
                                     target_warp_flow_all=target_wf_cur,
                                     s2t_corr_flow=s2t_cf)
            data_obj = 0.0001* wf_opt_weight['data_weight'] * (wf_cur[uv_valid_mask] - all_keep['warp_flow'][frame_idx])
            reg_obj = 0.1 * wf_opt_weight['reg_weight'] * (wf_cur[reg_idx[:, 0]] - wf_cur[reg_idx[:, 1]])

            ch.minimize({'wf_obj': wf_obj, 'data_obj': data_obj, 'reg_obj': reg_obj},
                        [wf_cur],
                        method='dogleg',
                        options={
                            'e_3': .01,
                        })

    with open('../data/optimal_warp_flow_level_00.pkl', 'wb') as f:
        pkl.dump(optimal_warp_flow_keep, f, pkl.HIGHEST_PROTOCOL)
    #with open('../data/optimal_warp_flow_level_00.pkl', 'rb') as f:
    #    optimal_warp_flow_opt_keep = pkl.load(f)
    optimal_warp_flow_opt_keep = optimal_warp_flow_keep

    '''
     estimate backward warp flow from forward warp flow
    '''
    optimal_warp_backflow_keep = []
    for frame_idx in range(frame_num):
        warp_flow = optimal_warp_flow_opt_keep[frame_idx].r
        uv_all = all_keep['uv_all'][frame_idx]
        warp_uv = np.round(uv_all + warp_flow).astype(np.int)
        warp_valid_mask = np.logical_and(np.logical_and(warp_uv[:, 0] >= 0, warp_uv[:, 0] < all_keep['width'] - 1),
                                         np.logical_and(warp_uv[:, 1] >= 0, warp_uv[:, 1] < all_keep['height'] - 1))

        # get valid backward warp prior
        back_uv = uv_all[warp_valid_mask]
        back_warp_uv = warp_uv[warp_valid_mask]
        back_flow = back_uv - back_warp_uv

        # construct data term
        u_min = np.maximum(np.min(back_warp_uv[:, 0]) - 20, 0)
        u_max = np.minimum(np.max(back_warp_uv[:, 0]) + 20, all_keep['width'] - 1) + 1
        v_min = np.maximum(np.min(back_warp_uv[:, 1]) - 20, 0)
        v_max = np.minimum(np.max(back_warp_uv[:, 1]) + 20, all_keep['height'] - 1) + 1
        back_warp_uv_all = np.hstack(
            (np.array(range(u_min, u_max)).reshape(1, -1).repeat(v_max - v_min, axis=0).reshape(-1, 1),
             np.array(range(v_min, v_max)).repeat(u_max - u_min).reshape(-1, 1)))

        tmp_mask = -np.ones((all_keep['height'], all_keep['width']), dtype=np.int)
        tmp_mask[back_warp_uv[:, 1], back_warp_uv[:, 0]] = range(back_warp_uv.shape[0])
        back_warp_valid_idx = tmp_mask[back_warp_uv_all[:, 1], back_warp_uv_all[:, 0]]
        back_warp_valid_mask = back_warp_valid_idx != -1
        back_warp_valid_idx = back_warp_valid_idx[back_warp_valid_mask]

        back_wf = ch.Ch(np.zeros((back_warp_uv_all.shape[0], 2), dtype=np.float))
        data_obj = back_wf[back_warp_valid_mask] - back_flow[back_warp_valid_idx]

        # construct reg term
        tmp_mask = -np.ones((all_keep['height'], all_keep['width']), dtype=np.int)
        tmp_mask[back_warp_uv_all[:, 1], back_warp_uv_all[:, 0]] = range(back_warp_uv_all.shape[0])
        ud_I1 = np.array(range(v_min, v_max - 1)).repeat(u_max - u_min)
        ud_I2 = np.array(range(v_min + 1, v_max)).repeat(u_max - u_min)
        ud_J = np.array(range(u_min, u_max)).reshape((1, -1)).repeat(v_max - v_min - 1, axis=0).ravel()
        lr_I = np.array(range(v_min, v_max)).repeat(u_max - u_min - 1)
        lr_J1 = np.array(range(u_min, u_max - 1)).reshape((1, -1)).repeat(v_max - v_min, axis=0).ravel()
        lr_J2 = np.array(range(u_min + 1, u_max)).reshape((1, -1)).repeat(v_max - v_min, axis=0).ravel()
        reg_idx = np.vstack(
            (np.hstack((tmp_mask[ud_I1, ud_J].reshape((-1, 1)), tmp_mask[ud_I2, ud_J].reshape((-1, 1)))),
             np.hstack((tmp_mask[lr_I, lr_J1].reshape((-1, 1)), tmp_mask[lr_I, lr_J2].reshape((-1, 1))))))
        assert(np.all(reg_idx[:, 0] != -1))
        assert(np.all(reg_idx[:, 1] != -1))
        reg_obj = back_wf[reg_idx[:, 0]] - back_wf[reg_idx[:, 1]]

        ch.minimize({'data_obj': data_obj, 'reg_obj': reg_obj},
                    [back_wf],
                    method='dogleg',
                    options={
                        'e_3': .01,
                    })

        # remove the meaningless part
        tmp_mask = np.zeros((all_keep['height'], all_keep['width']), dtype=np.bool)
        tmp_mask[uv_all[:, 1], uv_all[:, 0]] = True
        back_uv1 = np.round(back_warp_uv_all + back_wf.r).astype(np.int)
        back_uv1[:, 0] = np.maximum(np.minimum(all_keep['width'] - 1, back_uv1[:, 0]), 0)
        back_uv1[:, 1] = np.maximum(np.minimum(all_keep['height'] - 1, back_uv1[:, 1]), 0)
        back_valid = tmp_mask[back_uv1[:, 1], back_uv1[:, 0]]
        back_warp_uv_all = back_warp_uv_all[back_valid]
        back_wf = back_wf.r[back_valid]

        # display
        src_img = all_keep['color'][frame_idx]
        back_uv1 = np.round(back_warp_uv_all + back_wf).astype(np.int)
        back_uv1[:, 0] = np.maximum(np.minimum(all_keep['width'] - 1, back_uv1[:, 0]), 0)
        back_uv1[:, 1] = np.maximum(np.minimum(all_keep['height'] - 1, back_uv1[:, 1]), 0)
        warp_img = np.zeros_like(src_img, dtype=src_img.dtype)
        warp_img[back_warp_uv_all[:, 1], back_warp_uv_all[:, 0]] = src_img[back_uv1[:, 1], back_uv1[:, 0]]
        cv2.imshow('debug', warp_img)
        cv2.waitKey(1000)

        optimal_warp_backflow_keep.append({'back_warp_uv_all': back_warp_uv_all, 'back_warp_flow': back_wf})

    with open('../data/optimal_back_warp_flow_level_00.pkl', 'wb') as f:
        pkl.dump(optimal_warp_backflow_keep, f, pkl.HIGHEST_PROTOCOL)

    #with open('../data/optimal_back_warp_flow_level_00.pkl', 'rb') as f:
    #    optimal_warp_backflow_keep = pkl.load(f)

    '''
    write image and mask
    '''
    for frame_idx in range(frame_num):
        back_warp_uv_all = optimal_warp_backflow_keep[frame_idx]['back_warp_uv_all']
        back_wf = optimal_warp_backflow_keep[frame_idx]['back_warp_flow']

        src_img = all_keep['color'][frame_idx]
        back_uv1 = np.round(back_warp_uv_all + back_wf).astype(np.int)
        back_uv1[:, 0] = np.maximum(np.minimum(all_keep['width'] - 1, back_uv1[:, 0]), 0)
        back_uv1[:, 1] = np.maximum(np.minimum(all_keep['height'] - 1, back_uv1[:, 1]), 0)
        warp_img = np.zeros_like(src_img, dtype=src_img.dtype)
        warp_img[back_warp_uv_all[:, 1], back_warp_uv_all[:, 0]] = src_img[back_uv1[:, 1], back_uv1[:, 0]]
        cv2.imshow('debug', warp_img)
        cv2.waitKey(1000)

        warp_img = (warp_img * 255).astype(np.uint8)
        warp_img_mask = (np.sum(warp_img, axis=2) != 0).astype(np.uint8) * 255
        cv2.imwrite('../data/{:03d}.png'.format(frame_idx), warp_img)
        cv2.imwrite('../data/mask/{:03d}.png'.format(frame_idx), warp_img_mask)



    # for frame_idx in range(frame_num):
    #     warp_flow = optimal_warp_flow_keep[frame_idx].r
    #     warp_flow_opt = optimal_warp_flow_opt_keep[frame_idx].r
    #     uv_all = all_keep['uv_all'][frame_idx]
    #     warp_uv = np.round(uv_all + warp_flow).astype(np.int)
    #     warp_uv[:, 0] = np.maximum(np.minimum(all_keep['width'] - 1, warp_uv[:, 0]), 0)
    #     warp_uv[:, 1] = np.maximum(np.minimum(all_keep['height'] - 1, warp_uv[:, 1]), 0)
    #     warp_uv_opt = np.round(uv_all + warp_flow_opt).astype(np.int)
    #     warp_uv_opt[:, 0] = np.maximum(np.minimum(all_keep['width'] - 1, warp_uv_opt[:, 0]), 0)
    #     warp_uv_opt[:, 1] = np.maximum(np.minimum(all_keep['height'] - 1, warp_uv_opt[:, 1]), 0)
    #
    #     src_img = all_keep['color'][frame_idx]
    #     warp_img = np.zeros_like(src_img, dtype=src_img.dtype)
    #     warp_img_opt = np.zeros_like(src_img, dtype=src_img.dtype)
    #
    #     warp_img[warp_uv[:, 1], warp_uv[:, 0]] = src_img[uv_all[:, 1], uv_all[:, 0]]
    #     warp_img_opt[warp_uv_opt[:, 1], warp_uv_opt[:, 0]] = src_img[uv_all[:, 1], uv_all[:, 0]]
    #
    #     cv2.imshow('debug', warp_img)
    #     cv2.waitKey()
    #     cv2.imshow('debug_2', warp_img_opt)
    #     cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'pose_file',
        type=str,
        help="File that outputs from videoAvatar step 1")
    parser.add_argument(
        'shape_file',
        type=str,
        help="File that contains shape from videoAvatar step 2")
    parser.add_argument(
        'image_file',
        type=str,
        help="File that contains original video frames")
    parser.add_argument(
        'mask_file',
        type=str,
        help="File that contains original masks")
    parser.add_argument(
        'camera',
        type=str,
        help="pkl file that contains camera settings")
    parser.add_argument(
        '--model', '-m',
        default='vendor/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl',
        help='Path to SMPL model')

    args = parser.parse_args()

    main(args.pose_file, args.shape_file, args.image_file, args.mask_file, args.camera, args.model)
