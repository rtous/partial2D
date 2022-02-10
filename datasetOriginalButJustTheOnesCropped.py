
import json
import cv2
import numpy as np
import math
import poseUtils
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import traceback
import argparse
import os
import openPoseUtils
import sys
from shutil import copyfile

argv = sys.argv
try:
    INPUTPATH_CROPPED=argv[1]
    INPUTPATH_ORIGINAL=argv[2]
    OUTPUTPATH=argv[3]

except ValueError:
    print("Wrong arguments. Expecting two paths.")

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True)

copied = 0
scandirIterator = os.scandir(INPUTPATH_CROPPED)
for item in scandirIterator:
	filename = str(item.name)

	extension = os.path.splitext(filename)[1]
	if extension == ".json":
		filename_without_extension = os.path.splitext(filename)[0]
		indexUnderscore = filename.find('_')
		filenameoriginal = filename[:indexUnderscore]+".json"#+"_keypoints.json" 	
		sourcepath = join(INPUTPATH_ORIGINAL, filenameoriginal)
		targetpath = join(OUTPUTPATH, filenameoriginal)
		if not os.path.isfile(targetpath):
			copyfile(sourcepath, targetpath)
			copied += 1
			print("copied = ", copied)	
scandirIterator.close()




