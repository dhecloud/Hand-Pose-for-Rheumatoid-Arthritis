from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from REN import REN
from MSRADataset import MSRADataset, read_joints, read_depth_from_bin, get_center, _crop_image, _normalize_joints, data_rotate_chance, data_scale_chance, data_translate_chance, _unnormalize_joints
from main import draw_pose
import time
import argparse
import ctypes
import _ctypes
import sys
import numpy as np
import torch
import cv2
if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

parser = argparse.ArgumentParser(description='Main Applications')
parser.add_argument('--input_size', type=int, default=130, help='decay lr by 10 after _ epoches')
parser.add_argument('--num_joints', type=int, default=42, help='decay lr by 10 after _ epoches')
parser.add_argument('--checkpoint', type=str, default='experiments/exp_h/checkpoint.pth.tar', help='path/to/checkpoint.pth.tar')

class Display(object):
    def __init__(self, args):
        self.args= args
        self._done = False
        if self.args.input_size == 130:
            self.args.cuda = False
        else:
            self.args.cuda = torch.cuda.is_available()

        print('### Usage of CUDA (GPU):', self.args.cuda, '###')
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)



        #load model
        self.model = REN(args)
        self.model = self.load_checkpoint(self.args.checkpoint, self.model).double()
        print(self.args.checkpoint, 'loaded!')
        if self.args.cuda:
            self.model = self.model.cuda()
            # cudnn.benchmark=False



    def load_checkpoint(self, path, model):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])

        return model



    def draw_depth_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the infrared image. it works with Kinect studio though
            return
        target_surface.lock()
        f8=np.uint8(frame.clip(1,4000)/16.)
        frame8bit=np.dstack((f8,f8,f8))
        # print(frame8bit.shape)
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def run(self):
        torch.no_grad()
        self.model.eval()
        M = cv2.getRotationMatrix2D((320/2,240/2),270,1)
        # -------- Main Program Loop -----------
        while not self._done:

            if self._kinect.has_new_depth_frame():
                st = time.time()
                frame = self._kinect.get_last_depth_frame().astype(float).reshape(424,512)
                frame = frame[92:92+240,96:96+320]
                frame[frame<500] = 0
                frame[frame>700] = 0
                max = frame.max()
                min = frame.min()
                visualize = 255*((frame -min))/(max-min)
                cv2.imshow('DepthDisplay', visualize)
                center = np.array([160,120,600])
                depth = _crop_image(frame, center, self.args.input_size, False, 285.63, 285.63, 160,120)
                cv2.imshow('results', depth)
                depth = torch.tensor(depth).double()
                depth = depth.unsqueeze(0)
                depth = depth.unsqueeze(0)
                if self.args.cuda:
                    depth = depth.cuda()
                results = self.model(depth).reshape(21,2)
                tmp1 = np.zeros((21,3))
                for i in range(len(results)):
                    tmp1[i,:2] = results[i]
                results = _unnormalize_joints(tmp1,center, args.input_size, 285.63, 285.63, 160,120)
                canvas = np.ones((240,320))
                drawn = draw_pose(canvas, results)
                tt = time.time()-st
                # print('time taken:', tt)
                cv2.imshow('joints', drawn)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        self._kinect.close()


if __name__ == '__main__':
    args = parser.parse_args()
    display = Display(args);
    with torch.no_grad():
        display.run()
