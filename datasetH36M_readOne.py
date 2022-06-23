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
import json
#from spacepy import pycdf
import cdflib
import util_viz


'''
Read one file from H36M dataset. 
I used it to see how it deals with unknown joints
'''

#Before: tar -xvf /Volumes/ElementsDat/pose/H36M/H36M/2D/Poses_D2_Positions_S1.tgz 
#Before: pip install cdflib

INPUTPATH = "/Volumes/ElementsDat/pose/H36M/H36M/2D/S1/MyPoseFeatures/D2_Positions/"
MAX = 10

scandirIterator = os.scandir(INPUTPATH)
i = 0
for item in scandirIterator:
	if i >= MAX:
		break
	filename = str(item.name)
	print(filename)
	extension = os.path.splitext(filename)[1]
	if extension == ".cdf":
		path = join(INPUTPATH, filename)
		cdf = cdflib.CDF(path)
		#cdf = pycdf.CDF(path)
		print(cdf.cdf_info())
		pose = cdf.varget("Pose")
		print(pose[0][0])
		util_viz.visualizeOne(pose[0][0], "H36M", "borrar.png")
		sys.exit()
		i += 1
	    
