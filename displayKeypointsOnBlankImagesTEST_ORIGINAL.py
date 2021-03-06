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
import util_viz
import h36mUtils


WIDTH = 64
HEIGHT = 64
SPINESIZE = WIDTH/4
THRESHOLD = 0.1
HAVETHRESHOLD = True
#INPUTPATHIMAGES = "/Users/rtous/DockerVolume/charade/input/images"
#INPUTPATHKEYPOINTS = "/Users/rtous/DockerVolume/charade/results/openpose/2D_keypoints"
#OUTPUTPATH = "/Users/rtous/DockerVolume/charade/result/2D/blank_imagesOPENPOSE"

INPUTPATHKEYPOINTS = "/Users/rtous/DockerVolume/partial2D/dynamicData/H36Mtest"
OUTPUTPATH = "/Users/rtous/DockerVolume/partial2D/dynamicData/H36MtestImages"


pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

#Això per displayar les completades:
#PROBLEMA: Els json amb els keypoints només en tenen 15

jsonFiles = [f for f in listdir(INPUTPATHKEYPOINTS) if isfile(join(INPUTPATHKEYPOINTS, f)) and f.endswith(".json")]
for filename_keypoints in jsonFiles:
    filename_noextension = splitext(filename_keypoints)[0]
    #Obtain OpenPose keypoints (25 keypoints, list, with confidence)
    #keypoints, dummy, dummy, dummy = openPoseUtils.json2normalizedKeypoints(join(INPUTPATHKEYPOINTS, filename_keypoints))
    keypoints = openPoseUtils.json2Keypoints(join(INPUTPATHKEYPOINTS, filename_keypoints)) 
    keypoints, dummy = openPoseUtils.removeConfidence(keypoints)
    keypoints = [item for sublist in keypoints for item in sublist]
    keypoints = [float(k) for k in keypoints]
    keypoints = np.array(keypoints).flatten()

    #Add the missing 10 keypoints from openpose (as we work with 15 instead of 25)
    keypoints = np.pad(keypoints, (0, 20), 'constant')
 
    target_path = join(OUTPUTPATH, filename_noextension+"_pose1.jpg")
    util_viz.visualizeOne(keypoints, "OPENPOSE", target_path)

       
#Això per displayar les originals de CHARADE:
'''
imageFiles = [f for f in listdir(INPUTPATHIMAGES) if isfile(join(INPUTPATHIMAGES, f)) and (f.endswith(".png") or f.endswith(".jpg"))]
for filename_image in imageFiles:
    filename_noextension = splitext(filename_image)[0]
    filename_keypoints = filename_noextension+"_keypoints.json"
    #Obtain OpenPose keypoints (25 keypoints, list, with confidence)
    #keypoints, dummy, dummy, dummy = openPoseUtils.json2normalizedKeypoints(join(INPUTPATHKEYPOINTS, filename_keypoints))
    keypoints = openPoseUtils.json2Keypoints(join(INPUTPATHKEYPOINTS, filename_keypoints)) 
    keypoints, dummy = openPoseUtils.removeConfidence(keypoints)
    keypoints = [item for sublist in keypoints for item in sublist]
    keypoints = [float(k) for k in keypoints]
    keypoints = np.array(keypoints).flatten()




    #keypoints = openPoseUtils.json2KeypointsFlat(join(INPUTPATHKEYPOINTS, filename_keypoints))
    #keypoints, dummy = openPoseUtils.removeConfidence(keypoints)
    #keypoints = np.array(keypoints)
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

    #blackImage = np.zeros((orighinalImageHeight,originalImageWidht,3), dtype=np.uint8)
    #blackImage[:] = (255, 255, 255)

    #poseUtils.draw_pose(blackImage, keypoints, THRESHOLD, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, HAVETHRESHOLD, 8)
    
    target_path = join(OUTPUTPATH, filename_noextension+"_pose1.jpg")
    #H36Mkeypoints = h36mUtils.openpose2H36M(keypoints)
    #print(keypoints)
    #print(H36Mkeypoints)
    print(keypoints)

    #WARNING: POSE_BODY_25_BODY_PARTS_DICT should have 25 parts
    util_viz.visualizeOne(keypoints, "OPENPOSE", target_path)

    #cv2.imwrite(target_path, blackImage)
       
'''





    



