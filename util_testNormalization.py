from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytorchUtils
import argparse
from torchvision.datasets import MNIST
import pathlib
import json
from os import listdir
from os.path import isfile, join, splitext
import openPoseUtils
import poseUtils
import cv2
import traceback
import BodyModelOPENPOSE25
import BodyModelOPENPOSE15

bodyModel = BodyModelOPENPOSE25
NORMALIZATION = "center_scale"

print(bodyModel)
keypoints = openPoseUtils.json2Keypoints("dynamicData/012_keypoints.json")
print("original keypoints: ", keypoints)
poseUtils.debugKeypoints(keypoints)
imgWithKyepoints = cv2.imread("dynamicData/012.jpg")
poseUtils.draw_pose(imgWithKyepoints, keypoints, -1, bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)

cv2.imwrite("data/output/test_original.jpg", imgWithKyepoints)


##############
normalized_keypoints, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints("dynamicData/012_keypoints.json", bodyModel)
print("scaleFactor=",scaleFactor)
print("x_displacement=",x_displacement)
print("y_displacement=",y_displacement)
print("normalized keypoints: ", normalized_keypoints)
WIDTH = 200
HEIGHT = 200
background = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
background[:] = (255, 255, 255)
background = cv2.line(background, (int(WIDTH/2), 0), (int(WIDTH/2), HEIGHT-1), (0,0,255), 1)
background = cv2.line(background, (0, int(HEIGHT/2)), (WIDTH-1, int(HEIGHT/2)), (0,0,255), 1)
keypointsNP = poseUtils.keypoints2Numpy(normalized_keypoints)
poseUtils.draw_pose_scaled_centered(background, keypointsNP, -1, bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 1/200, centerX=WIDTH/2, centerY=HEIGHT/2, centerKeypointIndex=8)

cv2.imwrite("data/output/test_normalized.jpg", background)

##############
#denormalized_keypoints = openPoseUtils.denormalize(normalized_keypoints, scaleFactor, x_displacement, y_displacement)
denormalized_keypoints = openPoseUtils.denormalizeV2(normalized_keypoints, scaleFactor, x_displacement, y_displacement, normalizationMethod="center_scale")
print("denormalized_keypoints: ", denormalized_keypoints)
imgWithKyepoints = cv2.imread("dynamicData/012.jpg")
poseUtils.draw_pose(imgWithKyepoints, denormalized_keypoints, -1, bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
cv2.imwrite("data/output/test_denormalize.jpg", imgWithKyepoints)

background = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
background[:] = (255, 255, 255)
background = cv2.line(background, (int(WIDTH/2), 0), (int(WIDTH/2), HEIGHT-1), (0,0,255), 1)
background = cv2.line(background, (0, int(HEIGHT/2)), (WIDTH-1, int(HEIGHT/2)), (0,0,255), 1)
keypointsNP = poseUtils.keypoints2Numpy(denormalized_keypoints)
poseUtils.draw_pose_scaled_centered(background, keypointsNP, -1, bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 2, centerX=WIDTH/2, centerY=HEIGHT/2, centerKeypointIndex=8)
cv2.imwrite("data/output/test_denormalize_small.jpg", background)
