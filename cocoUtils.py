from pycocotools.coco import COCO
import poseUtils
import openPoseUtils
import numpy as np
import cv2
import sys
import os

#Check this for converting COCO and OpenPose: https://www.reddit.com/r/computervision/comments/bpx2us/conversion_from_coco_keypoints_to_open_pose/

#COCO VISIBILITY: 0 (not in the image, 1 (occluded), 2 (visible)

WIDTH = 128
HEIGHT = 128
SPINESIZE = WIDTH/6
HAVETHRESHOLD = True
THRESHOLD = -1

COCO_JOINTS = { 
    0:"Nose",
    1:"LEye",
    2:"REye",
    3:"LEar",
    4:"REar",
    5:"LShoulder",
    6:"RShoulder",
    7:"LElbow",
    8:"RElbow",
    9:"LWrist",
    10:"RWrist",
    11:"LHip",
    12:"RHip",
    13:"LKnee",
    14:"RKnee",
    15:"LAnkle",
    16:"RAnkle"
    #It has no neck, you can add it (pos 17) for drawing or for converting to openpose
} 

def geJointIndexFromName(jointName, jointNamesDict):
    idx = list(jointNamesDict.keys())[list(jointNamesDict.values()).index(jointName)]
    return idx       

#The way COCO draws it (without neck)
POSE_PAIRS_FOR_DRAWING = [
    [15,13],[13,11],[16,14],[14,12],[11,12],[5,11],[6,12],[5,6],
    [5,7],[6,8],[7,9],[8,10],[1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]
]

POSE_MPI_COLORS_RENDER_GPU = [
        [255,     0,    85], 
        [255,     0,     0], 
        [255,    85,     0], 
        [255,   170,     0], 
        [255,   255,     0], 
        [170,   255,     0], 
         [85,   255,     0], 
         [43,   255,     0], 
          [0,   255,     0], 
          [0,   255,    85], 
          [0,   255,   170], 
          [0,   255,   255],
          [0,   170,   255], 
          [0,    85,   255],
          [0,     0,   255], #borrar
          [0,     0,   255], #borrar
          [0,     0,   255], #borrar
          [0,     0,   255], #borrar
          [0,     0,   255], #borrar
          [0,     0,   255], #borrar
          [0,     0,   255], #borrar
          [0,     0,   255], #borrar
 ]

def computeMiddleJoint(keypoints, joint1name, joint2name):
    #Middle point between shoulders
    keypoint1 = keypoints[geJointIndexFromName(joint1name, COCO_JOINTS)]
    keypoint2 = keypoints[geJointIndexFromName(joint2name, COCO_JOINTS)]
    if keypoint1[2] > 0 and keypoint2[2] > 0:
        middle_keypoint_x = int((keypoint1[0]+keypoint2[0])/2.)
        middle_keypoint_y = int((keypoint1[1]+keypoint2[1])/2.)
        new_keypoint = (int(middle_keypoint_x), int(middle_keypoint_y), 1)
    else:
        new_keypoint = (0, 0, 0)
    return new_keypoint

def computeNeck(keypoints):
    return computeMiddleJoint(keypoints, "LShoulder", "RShoulder")

def computeMidHip(keypoints):
    return computeMiddleJoint(keypoints, "LHip", "RHip")

'''
def addNeck(keypoints):
    #In COCO the neck keypoint is the middle point between shoulders (and located in pos 1)
    new_keypoint = computeNeck(keypoints)
    
    keypoints_with_neck = []
    
    neck_pos = 17
    for j in range(0, neck_pos):
        keypoints_with_neck.append(keypoints[j])
    keypoints_with_neck.append(new_keypoint)
    for j in range(neck_pos, len(keypoints)):
        keypoints_with_neck.append(keypoints[j])    
    return keypoints_with_neck
'''


# function iterates ofver all ocurrences of a  person and returns relevant data row by row
def getCOCOAnnotations():
    train_annot_path = '/Volumes/ElementsDat/pose/COCO/ruben_structure/person_keypoints_train2017.json'
    coco = COCO(train_annot_path) # load annotations for training set
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # basic parameters of an image
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        # retrieve metadata for all persons in the current image
        anns = coco.loadAnns(ann_ids)

        yield [img_id, img_file_name, w, h, anns]

def coco2openpose(coco_keypoints):
    openpose_keypoints = []
    for i, openposeKeypointName in enumerate(openPoseUtils.POSE_BODY_25_BODY_PARTS):
        try:
            cocoIndex = geJointIndexFromName(openposeKeypointName, COCO_JOINTS)
            cocoKeypoint = coco_keypoints[cocoIndex]
            if cocoKeypoint[2] == 1 or cocoKeypoint[2] == 2:
                new_keypoint = (cocoKeypoint[0], cocoKeypoint[1], 1.0)
            else:
                new_keypoint = (0.0, 0.0, 0.0)
        except ValueError:
            if openposeKeypointName=="Neck":
                new_keypoint = computeNeck(coco_keypoints)
            elif openposeKeypointName=="MidHip":
                new_keypoint = computeMidHip(coco_keypoints)
            else:
                new_keypoint = (0,0,0)
        openpose_keypoints.append(new_keypoint)

    return openpose_keypoints



