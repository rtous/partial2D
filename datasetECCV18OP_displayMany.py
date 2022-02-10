
import json
import cv2
import numpy as np
import math
import poseUtils
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import os
import sys
from os import listdir
from os.path import isfile, join, splitext
import util_viz
import openPoseUtils


WIDTH = 64
HEIGHT = 64
THRESHOLD = 0.1
HAVETHRESHOLD = True
INPUTPATH = "dynamicData/H36M_ECCV18"

scandirIterator = os.scandir(INPUTPATH)
i = 0
MAX = 64
keypointsList = [None] * 64
for item in scandirIterator:
  if i >= MAX:
    break
  filename = str(item.name)
  print(filename)
  extension = os.path.splitext(filename)[1]
  if extension == ".json":
    try:
      keypoints = openPoseUtils.json2Keypoints(join(INPUTPATH, filename))
      #blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)
      #keypoints, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalize(keypoints, HAVETHRESHOLD)
      keypoints, dummy = openPoseUtils.removeConfidence(keypoints)
      keypointsNPflat = poseUtils.keypointsListFlatten(keypoints)
      keypointsList[i] = keypointsNPflat
      i += 1
    except:
      print("Not able to draw ", filename)

util_viz.visualizeMany(keypointsList, "OPENPOSE")


