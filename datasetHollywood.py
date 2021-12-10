
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

INPUTPATH = "dynamicData/H36M_ECCV18"
OUTPUTPATH = "data/H36M_ECCV18_HOLLYWOOD"

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True)

#parser = argparse.ArgumentParser()
#parser.add_argument('--path', type=str, default = "dynamicData/H36M_ECCV18_FILTERED_CROPPED")
#arguments, unparsed = parser.parse_known_args()
#Access it with arguments.path

jsonFiles = [f for f in listdir(INPUTPATH) if isfile(join(INPUTPATH, f))]
numOriginallyNotConfident = 0
numOK = 0
numDiscardedVariations = 0

numConfident = 0
sumRatioSpineNeck = 0
sumRatioSpineREye = 0
for filename in jsonFiles:
	filename_without_extension = os.path.splitext(filename)[0]
	try:
		keypoints = openPoseUtils.json2Keypoints(join(INPUTPATH, filename))

		#First discard not confident
		thresholdNoneBelow = 0.0
		thresholdNotMoreThanNBelow = 0.5
		N = 13
		if poseUtils.poseIsConfident(keypoints, thresholdNoneBelow, thresholdNotMoreThanNBelow, N):
			'''
			magnitudeSpine = openPoseUtils.magnitude_bone(keypoints, "Neck-MidHip")
			magnitudeNeck = openPoseUtils.magnitude_bone(keypoints, "Neck-Nose")
			magnitudeREye = openPoseUtils.magnitude_bone(keypoints, "Nose-REye")

			sumRatioSpineNeck += magnitudeSpine/magnitudeNeck
			sumRatioSpineREye += magnitudeSpine/magnitudeREye
			'''

			numConfident+= 1
			print("numConfident = ", numConfident) 

			#Now crop
			variations = openPoseUtils.crop(keypoints)
			numVariations = 0
			POTENTIAL_VARIATIONS = 6
			for i, v in enumerate(variations):
				openPoseUtils.keypoints2json(v, join(OUTPUTPATH, filename_without_extension+"_v"+str(i)+".json"))
				numOK += 1
				numVariations += 1
			numDiscardedVariations += POTENTIAL_VARIATIONS-numVariations
		else:
			numOriginallyNotConfident += 1

	except Exception as e:
		print("WARNING: Error reading ", filename)
        #print(e)
		traceback.print_exc()
print("numOK = ", numOK)
print("numOriginallyNotConfident = ", numOriginallyNotConfident)
print("numDiscardedVariations = ", numDiscardedVariations)
print("-----------------------------------------")
print("numConfident = ", numConfident)
#print("avgRatioSpineNeck = ", sumRatioSpineNeck/numConfident) 
#print("avgRatioSpineREye = ", sumRatioSpineREye/numConfident)






