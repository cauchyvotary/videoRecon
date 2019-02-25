#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import chumpy as ch
from opendr.renderer import ColoredRenderer, DepthRenderer, TexturedRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from opendr.simple import load_mesh
import numpy as np
import skimage.io as io
from chumpy.utils import row, col
import cv2

def render_image(model):

    v = model.v
    f = model.f
    vc = mesh.vc
    w, h = (512, 512)
    camera = ProjectPoints(v=v, rt=ch.zeros(3), t=ch.zeros(3),
                              f=ch.array([w, w]), c=ch.array([w, h]) / 2., k=ch.zeros(5))

    frustum = {'near': 0.1, 'far': 1000., 'width': w, 'height': h}
    rn = ColoredRenderer(camera=camera, v=v, f=f, frustum=frustum)
    rn.bgcolor = ch.zeros(3)
    rn.vc = LambertianPointLight(f=f, v=v, num_verts=len(v),light_pos=ch.array([0, 0, -1000]), vc=vc, light_color=ch.array([1., 1., 1.]))
    drn = DepthRenderer(camera=camera, v=v, f=f, frustum=frustum)

    return rn.r, drn.r

def render_image_texture(mesh):
    v = mesh.v
    f = mesh.f
    vc = mesh.vc
    ft = mesh.ft
    vt = mesh.vt
    texture = mesh.texture_image
    w, h = (1024/2, 1024/2)
    theta = np.pi/20
    #rotationx = np.array([[1, 0, 0], [0,np.cos(theta),np.sin(theta) ], [0, -np.sin(theta), np.cos(theta)]],
                        #dtype=np.float32)
    #rt = cv2.Rodrigues(rotationx)[0].flatten()
    #print('rt', rt)
    camera = ProjectPoints(v=v, rt=ch.zeros(3), t=ch.zeros(3),
                              f=ch.array([w, w])*2, c=ch.array([w, h]) / 2., k=ch.zeros(5))

    frustum = {'near': 0.1, 'far': 1000., 'width': w, 'height': h}
    trn = TexturedRenderer(camera=camera, v=v, f=f, frustum=frustum, vt=vt, ft=ft, texture_image=texture)
    trn.bgcolor = ch.zeros(3)
    #rn.vc = LambertianPointLight(f=f, v=v, num_verts=len(v),light_pos=ch.array([0, 0, -1000]), vc=vc, light_color=ch.array([1., 1., 1.]))
    trn.vc = np.ones_like(vc)
    return trn.r


mesh = load_mesh('/home/suoxin/Body/obj1/chenxin.obj')
mesh.v = np.asarray(mesh.v, order='C')
mesh.vc = mesh.v * 0 + ch.array([0.5,0.3,0.6])
mesh.v -= row(np.mean(mesh.v, axis=0))
mesh.v /= np.max(mesh.v)
mesh.v *= 2.0
rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]],dtype=np.float32)
rotation = rotation.dot(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]],dtype=np.float32))
print(rotation.shape)
rotation = ch.array(cv2.Rodrigues(rotation)[0])
mesh.v = mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
tmp = mesh.v

print(mesh.v.shape)
print(mesh.vt.shape)
for i in range(0, 30):
    print(i)
    theta = (i+1)/30.0 * (2*np.pi)
    rotation = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
    rotation = ch.array(cv2.Rodrigues(rotation)[0])
    trans = ch.array((0, 0,10))
    tmp1 = tmp.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
    mesh.v = tmp1 + row(np.asarray(trans))
    #image, depth_image = render_image(mesh)
    #depth_image = depth_image/np.max(depth_image)
    image = render_image_texture(mesh)

    #cv2.imwrite('/home/suoxin/Body/obj1/images2/'+str(i)+'.png', (image*255).astype(np.uint8))
    #cv2.imwrite('/home/suoxin/Body/obj1/depth_image/'+'depth'+str(i)+'.png', (depth_image*255).astype(np.uint8))
    cv2.imshow('image', image)
    io.imsave('/home/suoxin/Body/obj1/image/'+str(i)+'.png', (image*255).astype(np.uint8))
    cv2.waitKey()

'''
def read_mesh(path):
    mesh = load_mesh(path)
    mesh.v = np.asarray(mesh.v, order='C')
    mesh.vc = mesh.v * 0 + ch.array([0.5, 0.3, 0.6])
    mesh.v -= row(np.mean(mesh.v, axis=0))
    mesh.v /= np.max(mesh.v)
    mesh.v *= 2.0
    rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

    mesh.v = (rotation.dot(mesh.v.T)).T
    return mesh


def reder_dancing_model():
    path = '/home/suoxin/Body/obj2/dancing_model/'
    j = 0
    flag=0
    for i in range(0, 30):
        print(j)
        mesh = read_mesh(path + str(221+j) + '.obj')
        tmp = mesh.v

        theta = i/30.0 * (2*np.pi)
        rotation = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]],
                            dtype=np.float32)
        rotation = ch.array(cv2.Rodrigues(rotation)[0])
        trans = ch.array((0, 0, 7))

        tmp1 = tmp.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
        mesh.v = tmp1 + row(np.asarray(trans))

        image = render_image_texture(mesh)
        image1 = image.copy()
        image1[:,:,0] = image[:,:,2]
        image1[:,:,2] = image[:,:,0]
        #cv2.imshow('image',image1)
        #cv2.waitKey()
        if(j==9 and flag==0):
            flag=1
        elif(j==0 and flag==1):
            flag=0
        if(flag==0):
            j = (j+1)%10
        elif(flag==1):
            j = (j-1)%10
        import skimage.io as io
        io.imsave('/home/suoxin/Body/obj2/image/' + str(i)+'.png', (image*255).astype(np.uint8) )
'''
#reder_dancing_model()


