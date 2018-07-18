import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
from torchvision import transforms
import pickle
import sys
import time
import os

class MSRADataset(Dataset):
    def __init__(self, training=True):
        # transforms.RandomAffine(degrees = 0,translate=(-10,10))
        # self.transforms = transforms.Compose([
        #      transforms.RandomRotation(90),
        #      transforms.RandomHorizontalFlip()
        #   ])
        self.training = training
        if self.training:
            self.length = 1543059
            self.joints = read_joints(augment=True)
        else:
            self.length = 220500
            self.joints = read_joints([7],augment=True)
        # if training:
        #     self.images = read_MSRA(augment =False)
        #     print("Shape of training images: " + str(self.images.shape))
        #     self.joints = read_joints(augment=False)
        #     print("Shape of training joints: " + str(self.joints.shape))
        # else:
        #     self.images = (read_MSRA([7], augment =False))
        #     print("Shape of testing images: " + str(self.images.shape))
        #     self.joints = read_joints([7], augment =False)
        #     print("Shape of testing joints: " + str(self.joints.shape))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        file_index = np.floor(index/441)
        file_name = self._get_file_name(file_index)
        hf_index = index % 441
        with h5py.File(os.path.join("data_test",file_name), 'r') as hf:
            data = torch.tensor(hf['dataset_1'][hf_index:hf_index+1])

        return data, self.joints[index]

    def _get_file_name(self,file_index):
        assert(file_index>= 0)
        assert(file_index <3499)
        if self.training:
            if file_index <= 499:
                file_name = "0_" + str(int(file_index))
            elif file_index <= 999:
                file_name = "1_" + str(int(file_index-500))
            elif file_index <= 1499:
                file_name = "2_" + str(int(file_index-1000))
            elif file_index <= 1998:
                file_name = "3_" + str(int(file_index-1500))
            elif file_index <= 2498:
                file_name = "4_" + str(int(file_index-1999))
            elif file_index <= 2998:
                file_name = "5_" + str(int(file_index-2499))
            elif file_index <= 3498:
                file_name = "6_" + str(int(file_index-2999))
        else:
            file_name = "7_" + str(int(file_index))
        return file_name+".h5"


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

def read_MSRA(persons=[0,1,2,3,4,5,6], augment =False, pickle=False): #list of persons
    names = ['{:d}'.format(i).zfill(6) for i in range(500)]
    init = False
    init2 = False
    centers = []
    for person in persons:
        for name in names:
            # print(name)
            if ((person == 3) and (name == "000499")): #missing bin
                continue
            depth = read_depth_from_bin("data/P"+str(person)+"/5/"+name+"_depth.bin")

            #get centers
            center = get_center(depth)
            centers.append(center)

            #get cube and resize to 96x96
            depth = _crop_image(depth, center, is_debug=False)
            #normalize
            #depth = normalize(depth)
            assert not np.any(np.isnan(depth))
            assert ((depth>1).sum() == 0)
            assert ((depth<-1).sum() == 0)
            # augment_translate = augment_translation(depth)
            # augment_rotate = augment_rotation(depth)
            # augmented = augment_translate + augment_rotate
            depth = (torch.from_numpy(np.asarray(depth)))
            # print(depth.shape)
            depth = torch.unsqueeze(depth, 0)
            # print(depth.shape)
            if (not init):
                tmp = depth
                init = True
            else:
                tmp= torch.cat((tmp,depth),0)
        if (not init2):
            depth_images = tmp
            init2 = True
        else:
            depth_images= torch.cat((depth_images,tmp),0)
        init =False
    depth_images = torch.unsqueeze(depth_images, 1)
    if pickle:
        pickle.dump(depth_images, open(str(persons[0])+'.p', 'wb'))
    return depth_images

def read_MSRA_test(persons=[0,1,2,3,4,5,6], augment =False, pickleit=False): #list of persons
    names = ['{:d}'.format(i).zfill(6) for i in range(500)]
    init = False
    init2 = False
    centers = []
    for name in names:
        print(name)
        if ((int(persons[0]) == 3) and (name == "000499")): #missing bin
            continue
        depth = read_depth_from_bin("data/P"+str(persons[0])+"/5/"+name+"_depth.bin")

        #get centers
        center = get_center(depth)
        centers.append(center)

        #get cube and resize to 96x96
        depth = _crop_image(depth, center, is_debug=False)
        #normalize
        #depth = normalize(depth)
        assert not np.any(np.isnan(depth))
        assert ((depth>1).sum() == 0)
        assert ((depth<-1).sum() == 0)
        augment_translate = augment_translation(depth)
        # augment_rotate = augment_rotation(depth)
        # augmented = augment_translate + augment_rotate
        depth = (torch.from_numpy(np.asarray(augment_translate)))
        print("pickling..")
        print(depth.shape)
        depth = depth.numpy()
        with h5py.File("data_test/"+str(persons[0])+"_"+str(int(name))+'.h5', 'w') as h5f:
            h5f.create_dataset('dataset_1', data=depth)
        init = False



def read_joints(persons=[0,1,2,3,4,5,6], augment=False):
    joints = []
    for person in persons:
        with open("data/P"+str(person)+"/5/joint.txt") as f:
            num_joints = int(f.readline())
            for i in range(num_joints):
                joint = np.fromstring(f.readline(),sep=' ')
                joint = joint.reshape(21,3)
                joint = world2pixel(joint)
                joint = joint.reshape(63)
                joints.append(joint)

    joints = torch.from_numpy(np.asarray(joints))
    if augment:
        joints_augmented = []
        joints = joints.numpy()
        joints = joints.reshape(-1,21,3)
        for joint in joints:
            for i in range(-10, 11):    #left/right
                for j in range(-10, 11):    #up/down
                    tmp = joint
                    a = np.array([i, j, 0])
                    b = np.tile(a,(21,1))
                    tmp = tmp + b
                    joints_augmented.append(tmp)

        joints_augmented = (np.asarray(joints_augmented)).reshape(-1,63)
        return torch.from_numpy(joints_augmented)

    return joints


def augment_translation(depth):
    augment_translation = []
    rows,cols = depth.shape
    for i in range(-10, 11):
        for j in range(-10, 11):
            M = np.float32([[1,0,i],[0,1,j]])
            augment_translation.append(cv2.warpAffine(depth,M,(cols,rows)))
    return augment_translation

def augment_scaling(depth):
    augment_scale = []
    for i in [0.9, 1.0 ,1.1]:
        for j in [0.9, 1.0 ,1.1]:
            resize = cv2.resize(depth, (96,96), fx=i, fy=j)
            augment_scale.append(resize)

    return augment_scale

def augment_rotation(depth):
    augment_rotate = []
    rows,cols = depth.shape
    for i in range(-180,181):
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),i,1)
        augment_rotate.append(cv2.warpAffine(depth,M,(cols,rows)))
    return augment_rotate

def world2pixel(x):
    fx, fy, ux, uy = 241.42, 241.42, 160, 120
    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x

# read_MSRA_test(persons=[sys.argv[1]], pickleit=True)
# joints = read_joints([7],augment=True).numpy()
# assert ((joints>319).sum() == 0)
# print(joints[0].reshape(21,3)[0])
# print(joints[1].reshape(21,3)[0])
# print(joints[2].reshape(21,3)[0])
# print(joints[3].reshape(21,3)[0])
# print(joints[4].reshape(21,3)[0])
# joints = read_joints(augment=True)
# print(joints.shape)
# for root, dirs, files in os.walk("data_test", topdown=False):
#     for file in files:
#         print(os.path.join(root,file))
# with h5py.File("data_test/0_0.h5", 'r') as hf:
#     data = torch.tensor(hf['dataset_1'][0:1])
#     print(data.shape)
#     print(data)
#             if data.shape[0] != 441:
#                 print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#             total_size += data.shape[0]
# print(total_size)
