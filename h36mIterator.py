import json
#import cv2
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
import h36mUtils
import BodyModelOPENPOSE15
#import util_viz


#Before: cd /Volumes/ElementsDat/pose/H36M/H36M/2D
#Before: tar -xvf Poses_D2_Positions_S1.tgz
#Before: tar -xvf Poses_D2_Positions_S5.tgz
#Before: tar -xvf Poses_D2_Positions_S6.tgz
#Before: tar -xvf Poses_D2_Positions_S7.tgz
#Before: tar -xvf Poses_D2_Positions_S8.tgz
#Before: tar -xvf Poses_D2_Positions_S9.tgz
#Before: tar -xvf Poses_D2_Positions_S11.tgz

#Before: pip install cdflib

INPUTPATHS = [
"/2D/S1/MyPoseFeatures/D2_Positions/",
"/2D/S5/MyPoseFeatures/D2_Positions/",
"/2D/S6/MyPoseFeatures/D2_Positions/",
"/2D/S7/MyPoseFeatures/D2_Positions/",
"/2D/S8/MyPoseFeatures/D2_Positions/",
"/2D/S9/MyPoseFeatures/D2_Positions/",
"/2D/S11/MyPoseFeatures/D2_Positions/"
]
#OUTPUTPATH = "/Volumes/ElementsDat/pose/H36M/H36M/H36M"

#pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

CRED = '\033[91m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CBOLD     = '\33[1m'
CEND = '\033[0m'

def iterator(datasetPath, discardIncompletePoses=False):
	totalYield = 0
	for directory in INPUTPATHS:
		absoluteDirectory = datasetPath+directory
		print("Processing "+absoluteDirectory)
		scandirIterator = os.scandir(absoluteDirectory)
		for item in scandirIterator:
			#if files >= MAX:
			#    break
			filename = str(item.name)
			#print(filename)
			extension = os.path.splitext(filename)[1]
			if extension == ".cdf":
				path = join(absoluteDirectory, filename)
				cdf = cdflib.CDF(path)
				#print(cdf.cdf_info())
				pose = cdf.varget("Pose")
				#print(pose[0][0])
				#print("poses found in file: ", len(pose))
				for character in pose:
					intraframe = 0
					for keypointsH36M in character:
						#We will keep only one of each poses
						if intraframe % 10 == 0:
							keypoints=h36mUtils.h36m2openpose(keypointsH36M)  
							#thresholdNoneBelow = 0.0
							#thresholdNotMoreThanNBelow = 0.5
							#N = 10
							if discardIncompletePoses:
								#print(CRED + "ITERATOR: DISCARDING INCOMPLETE POSES" + CEND)
								thresholdNoneBelow = 0.01
								thresholdNotMoreThanNBelow = 0.01
								N = 0
							else:
								#print(CGREEN + "ITERATOR: NOT DISCARDING INCOMPLETE POSES" + CEND)
								thresholdNoneBelow = 0.0
								thresholdNotMoreThanNBelow = 0.5
								N = 10
							if poseUtils.poseIsConfident(keypoints, thresholdNoneBelow, thresholdNotMoreThanNBelow, N):
								referenceBoneIndex, referenceBoneSize = openPoseUtils.reference_bone(keypoints, BodyModelOPENPOSE15)
								magnitude_bone = openPoseUtils.magnitude_bone_from_index(keypoints, referenceBoneIndex, BodyModelOPENPOSE15)
								if magnitude_bone != 0:
									#print(intraframe)
									
									totalYield += 1
									yield keypoints
						intraframe += 1 
			#print("h36mIterator: yield ",totalYield)
		print("closing scandirIterator within h36mIterator")					
		scandirIterator.close()
	        

