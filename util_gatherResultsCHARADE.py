
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


INPUTPATHIMAGES = "/Users/rtous/DockerVolume/charade/input/images"

INPUTPATHKEYPOINTS = "/Users/rtous/DockerVolume/charade/results/openpose/2D_keypoints"
INPUTPATHKEYPOINTSIMAGES = "/Users/rtous/DockerVolume/charade/result/2D/blank_imagesOPENPOSE"
INPUTPATHKEYPOINTSIMAGESOVER = "/Users/rtous/DockerVolume/charade/result/2D/imagesOPENPOSE"

INPUTPATHKEYPOINTSFIXED = "/Users/rtous/DockerVolume/partial2D/data/output/H36M/CHARADE/keypoints"
INPUTPATHKEYPOINTSFIXEDIMAGES = "/Users/rtous/DockerVolume/partial2D/data/output/H36M/CHARADE/imagesBlank"
INPUTPATHKEYPOINTSFIXEDIMAGESOVER = "/Users/rtous/DockerVolume/partial2D/data/output/H36M/CHARADE/images"

OUTPUTPATH = "/Users/rtous/DockerVolume/charade/all/"

jsonFiles = [f for f in listdir(INPUTPATHKEYPOINTS) if f.endswith(".json") and isfile(join(INPUTPATHKEYPOINTS, f))]

for filename in jsonFiles:
	indexUnderscore = filename.rfind('_')
	filename_withoutsufix = filename[:indexUnderscore]
	subdir = join(OUTPUTPATH, filename_withoutsufix)
	pathlib.Path(subdir).mkdir(parents=True, exist_ok=True) 
	copyfile(join(INPUTPATHIMAGES, filename_withoutsufix+".png"), join(subdir, filename_withoutsufix+".jpg"))
	copyfile(join(INPUTPATHKEYPOINTS, filename), join(subdir, filename))
	copyfile(join(INPUTPATHKEYPOINTSIMAGES, filename_withoutsufix+"_pose.jpg"), join(subdir, filename_withoutsufix+"_pose.jpg"))
	copyfile(join(INPUTPATHKEYPOINTSIMAGESOVER, filename_withoutsufix+"_pose.jpg"), join(subdir, filename_withoutsufix+"_pose_over.jpg"))
	try:
		copyfile(join(INPUTPATHKEYPOINTSFIXED, filename), join(subdir, filename_withoutsufix+"_keypoints_fixed.json"))
		copyfile(join(INPUTPATHKEYPOINTSFIXEDIMAGES, filename_withoutsufix+"_pose.jpg"), join(subdir, filename_withoutsufix+"_pose_fixed.jpg"))
		copyfile(join(INPUTPATHKEYPOINTSFIXEDIMAGESOVER, filename_withoutsufix+"_pose.jpg"), join(subdir, filename_withoutsufix+"_pose_fixed_over.jpg"))
	
	except:

		print("WARNING: DISCARDED: "+join(INPUTPATHKEYPOINTSFIXED, filename))



	