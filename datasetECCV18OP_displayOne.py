
import json
import cv2
import numpy as np
import math
import poseUtils
import openPoseUtils
import pathlib
import traceback
import argparse
import os
import sys
from os import listdir
from os.path import isfile, join, splitext
import util_viz
import h36mUtils

WIDTH = 64
HEIGHT = 64
THRESHOLD = 0.1
HAVETHRESHOLD = True

keypoints = openPoseUtils.json2Keypoints('dynamicData/012_keypoints.json')

# With 3d-pose-baseline tools (directly openpose)
keypoints, dummy = openPoseUtils.removeConfidence(keypoints)
keypointsNPflat = poseUtils.keypointsListFlatten(keypoints)
util_viz.visualizeOne(keypointsNPflat, "OPENPOSE")


# With 3d-pose-baseline tools (converting to H36M)
'''
h36mKeypoints = h36mUtils.openpose2H36M(keypoints)
util_viz.visualizeOne(h36mKeypoints)
'''

#With my own drawing tools
'''
print(keypoints)
print(str(len(keypoints))+" keypoints found")
print(str(len(openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP))+" parts found")
blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)
keypoints, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalize(keypoints, HAVETHRESHOLD)
print(keypoints)
poseUtils.draw_pose(blank_image, keypoints, THRESHOLD, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, HAVETHRESHOLD)
poseUtils.displayImage(blank_image, WIDTH, HEIGHT)
'''





