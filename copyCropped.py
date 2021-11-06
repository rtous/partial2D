
import json
import cv2
import numpy as np
import math
import poseUtils
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import traceback
from shutil import copyfile

WIDTH = 64
HEIGHT = 64
SPINESIZE = WIDTH/4

HAVETHRESHOLD = True
REFERENCEPATH = "dynamicData/H36M_ECCV18_FILTERED_CROPPED"
INPUTPATH = "dynamicData/H36M_ECCV18"
OUTPUTPATH = "dynamicData/H36M_ECCV18_FILTERED"

jsonFiles = [f for f in listdir(REFERENCEPATH) if isfile(join(REFERENCEPATH, f))]

for filename in jsonFiles:
	copyfile(join(INPUTPATH, filename), join(OUTPUTPATH, filename))


	