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
"/Volumes/ElementsDat/pose/H36M/H36M/2D/S1/MyPoseFeatures/D2_Positions/",
"/Volumes/ElementsDat/pose/H36M/H36M/2D/S5/MyPoseFeatures/D2_Positions/",
"/Volumes/ElementsDat/pose/H36M/H36M/2D/S6/MyPoseFeatures/D2_Positions/",
"/Volumes/ElementsDat/pose/H36M/H36M/2D/S7/MyPoseFeatures/D2_Positions/",
"/Volumes/ElementsDat/pose/H36M/H36M/2D/S8/MyPoseFeatures/D2_Positions/",
"/Volumes/ElementsDat/pose/H36M/H36M/2D/S9/MyPoseFeatures/D2_Positions/",
"/Volumes/ElementsDat/pose/H36M/H36M/2D/S11/MyPoseFeatures/D2_Positions/"
]
OUTPUTPATH = "/Volumes/ElementsDat/pose/H36M/H36M/H36M"

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 


files = 0
characters = 0
poses = 0
numOK = 0
numDiscarded = 0
#MAX = 1000
for directory in INPUTPATHS:
    print("Processing "+directory)
    scandirIterator = os.scandir(directory)
    for item in scandirIterator:
        #if files >= MAX:
        #    break
        filename = str(item.name)
        print(filename)
        extension = os.path.splitext(filename)[1]
        if extension == ".cdf":
            path = join(directory, filename)
            cdf = cdflib.CDF(path)
            #print(cdf.cdf_info())
            pose = cdf.varget("Pose")
            #print(pose[0][0])
            #print("poses found in file: ", len(pose))
            for character in pose:
                for keypointsH36M in character:
                    poses += 1
                    keypoints=h36mUtils.h36m2openpose(keypointsH36M)  
                    thresholdNoneBelow = 0.0
                    thresholdNotMoreThanNBelow = 0.5
                    N = 13
                    if poseUtils.poseIsConfident(keypoints, thresholdNoneBelow, thresholdNotMoreThanNBelow, N):
                        openPoseUtils.keypoints2json(keypoints, OUTPUTPATH+"/"+str(numOK)+".json")
                        numOK += 1 
                    else:
                        numDiscarded +=1         
                characters += 1
                print("%d files, %d characters, %d poses, %d ok, %d discarded" % (files, characters, poses, numOK, numDiscarded))
        files += 1
    print("%d files, %d characters, %d poses" % (files, characters, files))
    scandirIterator.close()

print("TOTAL: %d files, %d characters, %d poses" % (files, characters, files)) 
print("Written "+str(numOK))
print("Discarded "+str(numDiscarded))
        

