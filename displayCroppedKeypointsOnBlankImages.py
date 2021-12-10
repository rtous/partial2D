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
INPUTPATHIMAGES = "/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG"
#INPUTPATHKEYPOINTS = "/Users/rtous/DockerVolume/charade/input/keypoints"
INPUTPATHKEYPOINTS = "data/H36M_ECCV18_HOLLYWOOD"
#OUTPUTPATH = "/Users/rtous/DockerVolume/charade/result/2D/blank_images"
OUTPUTPATH = "data/H36M_ECCV18_HOLLYWOOD_debug"

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

keypointsFiles = [f for f in listdir(INPUTPATHKEYPOINTS) if isfile(join(INPUTPATHKEYPOINTS, f)) and f.endswith(".json")]
for filename_keypoints in keypointsFiles:

    filename_noextension = splitext(filename_keypoints)[0]
    #filename_keypoints = filename_noextension+"_keypoints.json"

    keypoints = openPoseUtils.json2Keypoints(join(INPUTPATHKEYPOINTS, filename_keypoints))

    filename_image = filename_noextension[:5]+".jpg"
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
     






    



