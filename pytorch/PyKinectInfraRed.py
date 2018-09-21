from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from REN import REN
from MSRADataset import MSRADataset, read_joints, read_depth_from_bin, get_center, _crop_image, _normalize_joints, data_rotate_chance, data_scale_chance, data_translate_chance, _unnormalize_joints
import argparse
import ctypes
import _ctypes
import pygame
import sys
import numpy as np
import torch
import cv2
if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing different bodies
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                    pygame.color.THECOLORS["blue"],
                    pygame.color.THECOLORS["green"],
                    pygame.color.THECOLORS["orange"],
                    pygame.color.THECOLORS["purple"],
                    pygame.color.THECOLORS["yellow"],
                    pygame.color.THECOLORS["violet"]]

parser = argparse.ArgumentParser(description='Main Applications')
parser.add_argument('--input_size', type=int, default=96, help='decay lr by 10 after _ epoches')
parser.add_argument('--num_joints', type=int, default=42, help='decay lr by 10 after _ epoches')
parser.add_argument('--checkpoint', type=str, default=None, help='path/to/checkpoint.pth.tar')

class InfraRedRuntime(object):
    def __init__(self, args):
        pygame.init()
        self.args=args
        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect infrared frames, 8bit grey, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.depth_frame_desc.Width, self._kinect.depth_frame_desc.Height), 0, 24)
        # here we will store skeleton data
        self._bodies = None

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._kinect.infrared_frame_desc.Width, self._kinect.infrared_frame_desc.Height),
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Infrared")

        #load model
        self.model = REN(args)
        self.model.cuda()




    def load_checkpoint(model):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def draw_infrared_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the infrared image. it works with Kinect studio though
            return
        target_surface.lock()
        f8=np.uint8(frame.clip(1,4000)/16.)
        frame8bit=np.dstack((f8,f8,f8))
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def draw_depth_frame(self, frame, target_surface):
        if frame is None:  # some usb hub do not provide the infrared image. it works with Kinect studio though
            return
        target_surface.lock()
        f8=np.uint8(frame.clip(1,4000)/16.)
        # np.savetxt('frame.txt',f8, fmt='%f')
        frame8bit=np.dstack((f8,f8,f8))
        # print(frame8bit.shape)
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        # print(address)
        ctypes.memmove(address, frame8bit.ctypes.data, frame8bit.size)
        del address
        target_surface.unlock()

    def run(self):
        torch.no_grad()
        self.model.eval()
        M = cv2.getRotationMatrix2D((320/2,240/2),270,1)
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)


            # --- Getting frames and drawing
            # print(self._kinect.has_new_color_frame())
            # frame = self._kinect.get_last_color_frame()
            # x = 240, y= 320
            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                # print(frame)
                # print(frame.size)
                # print(frame.shape)
                # frame[frame<500]  = 99999
                # frame[frame>600]  = 99999
                # print(np.amin(frame))
                self.draw_depth_frame(frame, self._frame_surface)
                frame = None
                depth= pygame.surfarray.array2d(self._frame_surface).astype(np.float32)[96:96+320, 92:92+240]     #x = 512, y =424
                depth = cv2.flip(depth, 0)
                depth = cv2.warpAffine(depth,M,(320,240))[:200,:280]
                center = get_center(depth)
                depth = _crop_image(depth, center, self.args.input_size, False, 285.63, 285.63, 160,120)
                depth = torch.tensor(depth).cuda()
                depth = depth.unsqueeze(0)
                depth = depth.unsqueeze(0)
                # cv2.imshow('results', depth)
                results = self.model(depth)
                print(results)
                input()

            #size of screen is 512,424
            #draw red box
            pygame.draw.rect(self._frame_surface, pygame.color.THECOLORS["red"], [136, 132, 280, 200], 1)
            self._screen.blit(self._frame_surface, (0,0))
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 InfraRed"
args = parser.parse_args()
game =InfraRedRuntime(args);
game.run();
