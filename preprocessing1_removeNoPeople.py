
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

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default = "dynamicData/H36M_ECCV18_FILTERED_CROPPED")
arguments, unparsed = parser.parse_known_args()
#Access it with arguments.path

jsonFiles = [f for f in listdir(arguments.path) if isfile(join(arguments.path, f))]
numRemoved = 0
numOK = 0
for filename in jsonFiles:
	f = open(join(arguments.path, filename))
	try:
		data = json.load(f)
		 
		if len(data['people']) == 0:
			numRemoved += 1
			os.remove(join(arguments.path, filename))
		else:
			numOK += 1
	except Exception:
		traceback.print_exc()
	f.close()
print("Files ok: ", numOK)
print("Files removed: ", numRemoved)





