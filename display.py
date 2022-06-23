import json
import cv2
import numpy as np
import math
import poseUtils
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import openPoseUtils
import sys
from PIL import Image

THRESHOLD = 0.1
HAVETHRESHOLD = True

argv = sys.argv
try:
    INPUTPATHKEYPOINTS=argv[1]
    OUTPUTPATH=argv[2]
    INPUTPATHIMAGES=argv[3]
    OVER_IMAGE=int(argv[4])  
except ValueError:
    print("Wrong arguments. Expecting three paths.")

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

imageFiles = [f for f in listdir(INPUTPATHIMAGES) if isfile(join(INPUTPATHIMAGES, f)) and (f.endswith(".png") or f.endswith(".jpg"))]
for filename_image in imageFiles:
    originalImagePath = join(INPUTPATHIMAGES, filename_image)
    filename_noextension = splitext(filename_image)[0]
    filename_keypoints = filename_noextension+"_keypoints.json"
    keypointsPath = join(INPUTPATHKEYPOINTS, filename_keypoints)
    if isfile(originalImagePath) and isfile(keypointsPath):
      keypoints = openPoseUtils.json2Keypoints(keypointsPath)
      #print("Pose with %d keypoints", (len(keypoints)))
      originalImagePIL = Image.open(originalImagePath)
      open_cv_image = np.array(originalImagePIL) 
      originalImage = open_cv_image[:, :, ::-1].copy() # Convert RGB to BGR 
      #originalImage = cv2.imread(originalImagePath)
      orighinalImageHeight, originalImageWidht, channels = originalImage.shape 

      if OVER_IMAGE == 0:
        background = np.zeros((orighinalImageHeight,originalImageWidht,3), dtype=np.uint8)
        background[:] = (255, 255, 255)
      else:
        background = originalImage


      poseUtils.draw_pose(background, keypoints, THRESHOLD, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP_25, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, HAVETHRESHOLD, 2)
      
      target_path = join(OUTPUTPATH, filename_noextension+"_pose.jpg")
      cv2.imwrite(target_path, background)
     
    else:
      print("WARNING:"+keypointsPath+" does not exist.")  
      #exit(0)
    






    



