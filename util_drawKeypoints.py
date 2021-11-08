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

keypoints = openPoseUtils.json2Keypoints("dynamicData/012_keypoints.json")
print("original keypoints: ", keypoints)
imgWithKyepoints = cv2.imread("dynamicData/012.jpg")
poseUtils.draw_pose(imgWithKyepoints, keypoints, -1, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
cv2.imwrite("data/output/test_original.jpg", imgWithKyepoints)


normalized_keypoints, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints("dynamicData/012_keypoints.json")
print("scaleFactor=",scaleFactor)
print("x_displacement=",x_displacement)
print("y_displacement=",y_displacement)
print("normalized keypoints: ", normalized_keypoints)

denormalized_keypoints = openPoseUtils.denormalize(normalized_keypoints, scaleFactor, x_displacement, y_displacement)
print("denormalized_keypoints: ", denormalized_keypoints)
imgWithKyepoints = cv2.imread("dynamicData/012.jpg")
poseUtils.draw_pose(imgWithKyepoints, denormalized_keypoints, -1, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
cv2.imwrite("data/output/test_denormalize.jpg", imgWithKyepoints)

