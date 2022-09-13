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
import traceback
import BodyModelOPENPOSE15
import BodyModelOPENPOSE25

THRESHOLD = 0
HAVETHRESHOLD = False

argv = sys.argv
try:
    INPUTPATHKEYPOINTS=argv[1]
    OUTPUTPATH=argv[2]
    INPUTPATHIMAGES=argv[3]
    OVER_IMAGE=int(argv[4])  
    SCALE=int(argv[5]) 
    BODY_MODEL=eval(argv[6])

except ValueError:
    print("Wrong arguments. Expecting three paths.")

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

jsonFiles = [f for f in listdir(INPUTPATHKEYPOINTS) if isfile(join(INPUTPATHKEYPOINTS, f)) and (f.endswith(".json"))]
for filename in jsonFiles:
    filename_noextension = splitext(filename)[0]
    originalPath = join(INPUTPATHKEYPOINTS, filename)
    keypoints = openPoseUtils.json2Keypoints(originalPath)
  
    WIDTH = 500
    HEIGHT = 500
    background = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
    background[:] = (255, 255, 255)
      
    try:
      keypoints, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalize(keypoints, BODY_MODEL, keepConfidence=True)
      keypoints = openPoseUtils.denormalize(keypoints, 1, x_displacement, y_displacement, keepConfidence=True)
      keypointsNP = poseUtils.keypoints2Numpy(keypoints)
      keypointsNP = poseUtils.scale(keypointsNP, 0.01)
      keypointsNP, x_displacement, y_displacement = poseUtils.center_pose(keypointsNP, WIDTH/2, HEIGHT/2-100, 1)#8 midhip
      poseUtils.draw_pose(background, keypointsNP, THRESHOLD, BODY_MODEL.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, HAVETHRESHOLD, 2)
    except:
      traceback.print_exc()
      print("Skipping originalImagePath (probably no reference bone)")
  
    target_path = join(OUTPUTPATH, filename_noextension+"_pose.jpg")
    cv2.imwrite(target_path, background)
   
    
    






    



