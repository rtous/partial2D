
import json
import cv2
import numpy as np
import math
import poseUtils

WIDTH = 64
HEIGHT = 64
SPINESIZE = WIDTH/4
THRESHOLD = 0.1
HAVETHRESHOLD = True

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

# Opening JSON file
#f = open('00001_keypoints_LATESTOPENPOSEWINDOWS.json',)
f = open('dynamicData/012_keypoints.json',)
 
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

print(keypoints)

print(str(len(keypoints))+" keypoints found")

print(str(len(POSE_BODY_25_PAIRS_RENDER_GP))+" parts found")

blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)

boneSpineIndex = findPart('Nose')

poseUtils.normalize_pose(keypoints, POSE_BODY_25_PAIRS_RENDER_GP, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, HAVETHRESHOLD)

poseUtils.draw_pose(blank_image, keypoints, THRESHOLD, POSE_BODY_25_PAIRS_RENDER_GP, POSE_BODY_25_COLORS_RENDER_GPU, HAVETHRESHOLD)

'''
neckIndex = findPart("Nose")
#print("neckIndex="+neckIndex)
neckJoints = POSE_BODY_25_PAIRS_RENDER_GP[neckIndex]
draw_part(blank_image, keypoints[neckJoints[0]], keypoints[neckJoints[1]], POSE_BODY_25_COLORS_RENDER_GPU[neckIndex])
'''
poseUtils.displayImage(blank_image, WIDTH, HEIGHT)

# Closing file
f.close()




