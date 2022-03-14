
import json
import cv2
import numpy as np
import math
import poseUtils
from os import listdir
from os.path import isfile, join, splitext
import pathlib
import traceback

'''
Specification of 15 baseline joints.
They are the first 15 openpose joints (from 25) 
That are also H36M joints
'''

JOINTS_15 = [
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
    "LAnkle"
]


JOINTS_DICT_15 = {
    0:"Nose",
    1:"Neck",
    2:"RShoulder",
    3:"RElbow",
    4:"RWrist",
    5:"LShoulder",
    6:"LElbow",
    7:"LWrist",
    8:"MidHip",
    9:"RHip",
    10:"RKnee",
    11:"RAnkle",
    12:"LHip",
    13:"LKnee",
    14:"LAnkle"
}

BONES_15 = [[1,8],   [1,2],   [1,5],   [2,3],   [3,4],   [5,6],   [6,7],   [8,9],   [9,10],  [10,11], [8,12],  [12,13], [13,14],  [1,0]]

BONE_NAMES_15 = ["Neck-MidHip", "Neck-RShoulder", "Neck-LShoulder", "RShoulder-RElbow", "RElbow-RWrist", "LShoulder-LElbow", "LElbow-LWrist", "MidHip-RHip", "RHip-RKnee", "RKnee-RAnkle", "MidHip-LHip", "LHip-LKnee", "LKnee-LAnkle", "Neck-Nose", "Nose-REye", "REye-REar", "Nose-LEye", "LEye-LEar", "LAnkle-LBigToe", "LBigToe-LSmallToe", "LAnkle-LHeel", "RAnkle-RBigToe", "RBigToe-RSmallToe", "RAnkle-RHeel"]

def first15Keypoints(keypoints):
    resulting_keypoints = []
    for i in range(15):
        resulting_keypoints.append(keypoints[i])
    return resulting_keypoints 