#requires dependencies from partial2D

import json
import cv2
import numpy as np
import math
import poseUtils
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import openPoseUtils
from PIL import Image


WIDTH = 64
HEIGHT = 64
SPINESIZE = WIDTH/4
THRESHOLD = 0.1
HAVETHRESHOLD = True
INPUTPATHIMAGES = "/Users/rtous/DockerVolume/charade/input/images"
INPUTPATHKEYPOINTS = "/Users/rtous/DockerVolume/charade/input/keypoints"
OUTPUTPATH = "/Users/rtous/DockerVolume/charade/result/2D/blank_images"

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

imageFiles = [f for f in listdir(INPUTPATHIMAGES) if isfile(join(INPUTPATHIMAGES, f)) and f.endswith(".png")]
for filename_image in imageFiles:

    filename_noextension = splitext(filename_image)[0]
    filename_keypoints = filename_noextension+"_keypoints.json"

    keypoints = openPoseUtils.json2Keypoints(join(INPUTPATHKEYPOINTS, filename_keypoints))

    originalImagePath = join(INPUTPATHIMAGES, filename_image)

    if isfile(originalImagePath):
      originalImagePIL = Image.open(originalImagePath)
      open_cv_image = np.array(originalImagePIL) 
      originalImage = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR 
      #originalImage = cv2.imread(originalImagePath)
    else:
      print(originalImagePath+" does not exist.")  
      exit(0)
    orighinalImageHeight, originalImageWidht, channels = originalImage.shape 

    blackImage = np.zeros((orighinalImageHeight,originalImageWidht,3), dtype=np.uint8)
    blackImage[:] = (255, 255, 255)

    poseUtils.draw_pose(blackImage, keypoints, THRESHOLD, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, HAVETHRESHOLD, 8)
    
    target_path = join(OUTPUTPATH, filename_noextension+"_pose.jpg")
    cv2.imwrite(target_path, blackImage)
     






    



