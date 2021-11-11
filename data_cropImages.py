
import json
import cv2
import numpy as np
import math
import poseUtils
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import traceback

#INPUTPATH = "data/H36M_ECCV18/images"
#OUTPUTPATH = "data/H36M_ECCV18/images_cropped"

INPUTPATH = "data/H36M_ECCV18/Test/IMG"
OUTPUTPATH = "data/H36M_ECCV18/Test/IMG_CROPPED"

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

files = [f for f in listdir(INPUTPATH) if isfile(join(INPUTPATH, f))]

for filename in files:
	img = cv2.imread(join(INPUTPATH, filename))
	height, width, channels = img.shape
	cv2.rectangle(img, (0, int(height/2.2)), (width, height), (0,0,0), -1)
	cv2.imwrite(join(OUTPUTPATH, filename), img)




