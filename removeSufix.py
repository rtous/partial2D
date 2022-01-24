
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


INPUTPATH = "dynamicData/H36M_ECCV18"
OUTPUTPATH = "/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_no_sufix"

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

jsonFiles = [f for f in listdir(INPUTPATH) if f.endswith(".json") and isfile(join(INPUTPATH, f))]

for filename in jsonFiles:
	indexUnderscore = filename.find('_')
	filename_withoutsufix = filename[:indexUnderscore]+".json"  
	copyfile(join(INPUTPATH, filename), join(OUTPUTPATH, filename_withoutsufix))


	