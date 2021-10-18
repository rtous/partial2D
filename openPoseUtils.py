
import json
import cv2
import numpy as np
import math
import poseUtils
from os import listdir
from os.path import isfile, join, splitext
import pathlib


WIDTH = 64
HEIGHT = 64
SPINESIZE = WIDTH/4
THRESHOLD = 0.1
HAVETHRESHOLD = True
INPUTPATH = "dynamicData/H36M_ECCV18"
OUTPUTPATH = "data/H36M_ECCV18/poselets"

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

#https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1644

#25 May 2019
#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/pose/poseParametersRender.hpp
 
POSE_BODY_25_BODY_PARTS = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    "Background"
]

POSE_BODY_25_COLORS_RENDER_GPU =[
        [255,     0,    85],
        [255,     0,     0], 
        [255,    85,     0], 
        [255,   170,     0],
        [255,   255,     0],
        [170,   255,     0],
         [85,   255,     0],
          [0,   255,     0],
        [255,     0,     0],
          [0,   255,    85],
          [0,   255,   170],
          [0,   255,   255],
          [0,   170,   255],
          [0,    85,   255],
          [0,     0,   255],
        [255,     0,   170],
        [170,     0,   255],
        [255,     0,   255],
         [85,     0,   255],
          [0,     0,   255],
          [0,     0,   255],
          [0,     0,   255],
          [0,   255,   255],
          [0,   255,   255],
          [0,   255,   255]
]

POSE_BODY_25_PAIRS_RENDER_GP = [[1,8],   [1,2],   [1,5],   [2,3],   [3,4],   [5,6],   [6,7],   [8,9],   [9,10],  [10,11], [8,12],  [12,13], [13,14],  [1,0],   [0,15], [15,17],  [0,16], [16,18],   [14,19],[19,20],[14,21], [11,22],[22,23],[11,24]]

def findPart(partName):
    for i, part in enumerate(POSE_BODY_25_BODY_PARTS):
        if part == partName:
            return i
    raise RuntimeError("part name not found:"+partName)

def json2normalizedKeypoints(path):

    # Opening JSON file
    f = open(path)
     
    # returns JSON object as
    # a dictionary
    data = json.load(f)
     
    # Iterating through the json
    # list
    person = data['people'][0]

    keypointsFlat = person['pose_keypoints_2d']

    #keypointsFlat = list(map(int, keypointsFlat))

    keypoints = list(zip(
        list(map(int, keypointsFlat[0::3])), 
        list(map(int, keypointsFlat[1::3])), 
        list(map(float, keypointsFlat[2::3]))  
    ))


    boneSpineIndex = findPart('Nose')

    poseUtils.normalize_pose(keypoints, POSE_BODY_25_PAIRS_RENDER_GP, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, HAVETHRESHOLD)

    f.close()

    return keypoints

def normalize(keypoints):

    boneSpineIndex = findPart('Nose')

    poseUtils.normalize_pose(keypoints, POSE_BODY_25_PAIRS_RENDER_GP, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, False)

    return keypoints

def removeConfidence(keypoints):
    for i, k in enumerate(keypoints):
        new_keypoint = (k[0], k[1])
        keypoints[i] = new_keypoint
    return keypoints

