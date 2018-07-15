from multiprocessing import Process
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
from torchvision import transforms
import pickle
import sys
class MSRADataset(Dataset):
    def __init__(self, training=True):
        # transforms.RandomAffine(degrees = 0,translate=(-10,10))
        # self.transforms = transforms.Compose([
        #      transforms.RandomRotation(90),
        #      transforms.RandomHorizontalFlip()
        #   ])

        if training:
            self.images = read_MSRA([0],augment =True)
            print("Shape of training images: " + str(self.images.shape))
            self.joints = read_joints([0],augment =True)
            print("Shape of training joints: " + str(self.joints.shape))
        else:
            self.images = (read_MSRA([7], augment =False))
            print("Shape of testing images: " + str(self.images.shape))
            self.joints = read_joints([7], augment =False)
            print("Shape of testing joints: " + str(self.joints.shape))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # if self.transforms is not None:
        #     self.images = self.transforms(self.images)
        return self.images[index], self.joints[index]

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
            print(name)
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

def read_joints(persons=[0,1,2,3,4,5,6], augment=False):
    joints = []
    for person in persons:
        with open("data/P"+str(person)+"/5/joint.txt") as f:
            num_joints = int(f.readline())
            for i in range(num_joints):
                if augment:
                    tmp = np.fromstring(f.readline(),sep=' ')
                    for i in range(802):
                        joints.append(tmp)
                else:
                    joints.append(np.fromstring(f.readline(),sep=' '))

    joints = torch.from_numpy(np.asarray(joints))


    return joints

def normalize(array):
    min = np.min(array)
    max = np.max(array)
    array = (2 * ((array - min)/(max - min))) -1

    return array

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


read_MSRA(persons=[sys.argv[1]], pickle=True)
