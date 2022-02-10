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
import csv

'''
Read one file from ECCV18 dataset. It contains images and 3D keypoints (not it is a 3D estimation challenge)
'''

INPUTPATH = "/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/POSE/"
MAX = 1

scandirIterator = os.scandir(INPUTPATH)
i = 0
for item in scandirIterator:
	if i >= MAX:
		break
	filename = str(item.name)
	print(filename)
	extension = os.path.splitext(filename)[1]
	if extension == ".csv":
		path = join(INPUTPATH, filename)
		with open(path) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			for row in csv_reader:
				print(row)
		i += 1
	    