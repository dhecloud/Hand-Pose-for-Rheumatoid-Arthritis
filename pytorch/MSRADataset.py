import cv2
import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import transforms
import pickle
import sys
import time
import os
import random

class MSRADataset(Dataset):
    def __init__(self, training=True, augment = False):

        self.training = training
        self.augment = augment
        if self.training:
            self.joints, self.keys = read_joints()
            self.length = len(self.joints)
        else:
            self.joints,self.keys = read_joints([8])
            self.length = len(self.joints)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        joint = self.joints[index]

        if self.training:
            person = self.keys[index][0]
            name = self.keys[index][1]
            file = '%06d' % int(self.keys[index][2])
            depth = read_depth_from_bin("data/P"+str(person)+"/"+str(name)+"/"+str(file)+"_depth.bin")

        else:
            person = self.keys[index][0]
            name = self.keys[index][1]
            file = '%06d' % int(self.keys[index][2])
            depth = read_depth_from_bin("data/P"+str(person)+"/"+str(name)+"/"+str(file)+"_depth.bin")

        center = get_center(depth)
        depth = _crop_image(depth, center, is_debug=False)
        assert not np.any(np.isnan(depth))
        assert ((depth>1).sum() == 0)
        assert ((depth<-1).sum() == 0)

        if self.augment:
            depth, joint = data_scale_chance(depth,joint, p=0.6)
            depth,joint = data_translate_chance(depth,joint, p = 0.6)
            depth,joint = data_rotate_chance(depth,joint, p = 0.6)

        data = torch.tensor(np.asarray(depth))
        data = data.unsqueeze(0)
        joint = torch.tensor(joint)

        return data, joint

def get_center(img, upper=1000, lower=10):
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    return centers

def _crop_image(img, center, is_debug=False):
    _fx, _fy, _ux, _uy = 241.42, 241.42, 160, 120
    _cube_size = 150
    _input_size = 96
    xstart = center[0] - _cube_size / center[2] * _fx
    xend = center[0] + _cube_size / center[2] * _fx
    ystart = center[1] - _cube_size / center[2] * _fy
    yend = center[1] + _cube_size / center[2] * _fy
    src = [(xstart, ystart), (xstart, yend), (xend, ystart)]
    dst = [(0, 0), (0, _input_size - 1), (_input_size - 1, 0)]
    trans = cv2.getAffineTransform(np.array(src, dtype=np.float32),
            np.array(dst, dtype=np.float32))
    res_img = cv2.warpAffine(img, trans, (_input_size, _input_size), None,
            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, center[2] + _cube_size)
    res_img -= center[2]
    res_img = np.maximum(res_img, -_cube_size)
    res_img = np.minimum(res_img, _cube_size)
    res_img /= _cube_size

    if is_debug:
        img_show = (res_img + 1) / 2;
        hehe = cv2.resize(img_show, (512, 512))
        cv2.imshow('debug', img_show)
        ch = cv2.waitKey(0)
        if ch == ord('q'):
            exit(0)
    return res_img

def read_depth_from_bin(image_name):
    f = open(image_name, 'rb')
    data = np.fromfile(f, dtype=np.uint32)
    width, height, left, top, right , bottom = data[:6]
    depth = np.zeros((height, width), dtype=np.float32)
    f.seek(4*6)
    data = np.fromfile(f, dtype=np.float32)
    depth[top:bottom, left:right] = np.reshape(data, (bottom-top, right-left))
    return depth

def data_translate_chance(depth, joint, p = 0.5):
    roll = random.uniform(0,1)
    if roll < p:
        rows,cols = depth.shape
        i = random.randint(-10,10)
        j = random.randint(-10,10)
        M = np.float32([[1,0,i],[0,1,j]])
        depth = cv2.warpAffine(depth,M,(cols,rows))

        joint = joint.numpy().reshape(21,3)
        tmp = joint
        a = np.array([i, j, 0])
        b = np.tile(a,(21,1))
        tmp = tmp + b
        joint = tmp.reshape(63)

    return depth, joint

def get_rotated_points(joints, M):
    for i in range(len(joints)):
        x = joints[i][0]
        y = joints[i][1]
        joints[i][0] = M[0,0]*x+ M[0,1]*y + M[0,2]
        joints[i][1] = M[1,0]*x + M[1,1]*y + M[1,2]

    return joints

def data_rotate_chance(depth, joint, p = 0.5):
    roll = random.uniform(0,1)
    if roll < p:
        angle = random.randint(-180,180)
        rows,cols = depth.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        joints = get_rotated_points(joint.reshape(21,3), M).reshape(63)

        depth = cv2.warpAffine(depth, M, (cols,rows))

    return depth, joint

def data_scale_chance(depth,joint,p=0.8):
    roll = random.uniform(0,1)
    if roll < p:
        if random.random() >= 0.5:
            depth, joint = data_zoom_chance(depth,joint)
        else:
            depth, joint = data_shrink_chance(depth,joint)
    return depth,joint

def data_shrink_chance(depth, joint):
    scale = [0.9,0.96,1]
    joint = joint.reshape(21,3)
    scale = scale[random.randint(0,2)]

    #resize/shrink
    depth = cv2.resize(depth,None,fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    joint[:,:2] = joint.reshape(21,3)[:,:2] * scale

    #pad
    rows,cols = depth.shape
    pad_t = pad_b = int((96 - int(rows))/2)
    pad_l = pad_r = int((96 - int(cols))/2)
    depth = cv2.copyMakeBorder(depth,pad_t,pad_b,pad_l,pad_r,cv2.BORDER_REPLICATE)
    assert(depth.shape == (96,96))
    joint[:,:1] = joint[:,:1] + (pad_l)
    joint[:,1:2] = joint[:,1:2] + (pad_t)
    joint = joint.reshape(63)
    return depth, joint

def data_zoom_chance(depth, joint):
    scale = [1, 1.04, 1.1]
    scale = scale[random.randint(0,2)]
    joint = joint.reshape(21,3)
    #resize/shrink
    depth = cv2.resize(depth,None,fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    joint[:,:2] = joint.reshape(21,3)[:,:2] * scale

    #trim
    rows,cols = depth.shape
    pad_t = pad_b = int((int(rows -96))/2)
    pad_l = pad_r = int((int(cols) - 96)/2)
    depth = depth[pad_t:rows-pad_b, pad_l:cols-pad_r]
    assert(depth.shape == (96,96))
    joint[:,:1] = joint[:,:1] - (pad_l)
    joint[:,1:2] = joint[:,1:2] - (pad_t)
    joint = joint.reshape(63)
    return depth, joint

def read_joints(persons=[0,1,2,3,4,5,6,7], poses= ["1","2","3","4",'5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']):
    joints = []
    index = 0
    keys = {}
    for person in persons:
        for pose in poses:
            with open("data/P"+str(person)+"/"+str(pose)+"/joint.txt") as f:
                num_joints = int(f.readline())
                for i in range(num_joints):
                    joint = np.fromstring(f.readline(),sep=' ')
                    joint = joint.reshape(21,3)
                    joint = world2pixel(joint)
                    joint = joint.reshape(63)
                    joints.append(joint)
                    keys[index]= [person,pose,i]
                    index +=1

    joints = torch.from_numpy(np.asarray(joints))

    return joints, keys

def world2pixel(x):
    x[:, 1] = x[:, 1] * -1
    x[:, 2] = x[:, 2] * -1
    fx, fy, ux, uy = 241.42, 241.42, 160, 120
    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x


msra = MSRADataset(training=True,augment = True)
for i in range(1000):
    msra.__getitem__(i)[0].shape
