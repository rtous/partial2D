
import json
import cv2
import numpy as np
import math
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import traceback
from shutil import copyfile
import os


INPUTPATH = "/Users/rtous/DockerVolume/partial2D/dynamicData/H36Mtest_original_v2_noreps"
OUTPUTPATH = "/Users/rtous/DockerVolume/partial2D/dynamicData/H36Mtest_v2_3D/original_noreps"

#pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

jsonFiles = [f for f in listdir(INPUTPATH) if f.endswith(".json") and isfile(join(INPUTPATH, f))]

for filename in jsonFiles:
	filenameNoExtension = os.path.splitext(filename)[0]
	targetFilename = filenameNoExtension+"_keypoints.json"
	copyfile(join(INPUTPATH, filename), join(OUTPUTPATH+"/keypoints", targetFilename))

	blankimage = np.zeros((1000,1000,3), np.uint8)
	cv2.imwrite(join(OUTPUTPATH+"/images", filenameNoExtension+".jpg"), blankimage)

	