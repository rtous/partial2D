import poseUtils
import openPoseUtils
import numpy as np
import sys
import os

# info here: https://github.com/qxcv/pose-prediction/blob/master/H36M-NOTES.md

H36M_JOINTS = { 
    0:"MidHip",
    1:"RHip",
    2:"RKnee",
    3:"RAnkle",
    4:"RightToeBase", #NOT STANDARD
    5:"SiteRightToeBase", #NOT STANDARD
    6:"LHip",
    7:"LKnee",
    8:"LAnkle",
    9:"LeftToeBase", #NOT STANDARD
    10:"SiteLeftToeBase", #NOT STANDARD
    11:"Spine",  #NOT STANDARD
    12:"Spine1", #NOT STANDARD
    13:"Neck",
    14:"Nose",
    15:"SiteHead",
    16:"LeftShoulder",#NOT STANDARD
    17:"LShoulder",#NOT STANDARD
    18:"LElbow",#NOT STANDARD
    19:"LWrist",#NOT STANDARD
    20:"LeftHandThumb",#NOT STANDARD
    21:"SiteLeftHandThumb",#NOT STANDARD
    22:"L_Wrist_End",#NOT STANDARD
    23:"SiteL_Wrist_End",#NOT STANDARD
    24:"RightShoulder",#NOT STANDARD
    25:"RShoulder",
    26:"RElbow",
    27:"RWrist",
    28:"RightHandThumb",#NOT STANDARD
    29:"SiteRightHandThumb",#NOT STANDARD
    30:"R_Wrist_End",#NOT STANDARD
    31:"SiteR_Wrist_End"#NOT STANDARD
}


def computeSpine(keypoints):
    return poseUtils.computeMiddleJoint(keypoints, "MidHip", "Neck", openPoseUtils.POSE_BODY_25_BODY_PARTS_DICT)

def openpose2H36M(openpose_keypoints):
    h36m_keypoints = []
    for key in H36M_JOINTS:
        try:
            h36mKeypointName= H36M_JOINTS[key]
            openposeIndex = poseUtils.geJointIndexFromName(h36mKeypointName, openPoseUtils.POSE_BODY_25_BODY_PARTS_DICT)
            openposeKeypoint = openpose_keypoints[openposeIndex]
            new_keypoint = (openposeKeypoint[0], openposeKeypoint[1])
        except ValueError:
            if h36mKeypointName=="Spine1":
                new_keypoint = computeSpine(openpose_keypoints)
                new_keypoint = (new_keypoint[0], new_keypoint[1])
            else:
                new_keypoint = (0,0)

        h36m_keypoints.append(new_keypoint)

    print(h36m_keypoints)

    h36m_keypoints_np = np.vstack(h36m_keypoints)

    h36m_keypoints = h36m_keypoints_np.flatten()

    print(h36m_keypoints)

    return h36m_keypoints

def h36m2openpose(h36m_keypoints_flat):
    h36m_keypoints = list(zip(
        list(map(int, h36m_keypoints_flat[0::2])), 
        list(map(int, h36m_keypoints_flat[1::2]))
    ))
    resulting_keypoints = []
    for key in openPoseUtils.POSE_BODY_25_BODY_PARTS_DICT:
        try:
            resultingKeypointName= openPoseUtils.POSE_BODY_25_BODY_PARTS_DICT[key]
            h36mIndex = poseUtils.geJointIndexFromName(resultingKeypointName, H36M_JOINTS)
            h36mKeypoint = h36m_keypoints[h36mIndex]
            new_keypoint = (h36mKeypoint[0], h36mKeypoint[1], 1.0)
        except ValueError:
            #print("Cannot find keypoint ", resultingKeypointName)
            #if resultingKeypointName=="Spine1":
            #    new_keypoint = computeSpine(openpose_keypoints)
            #    new_keypoint = (new_keypoint[0], new_keypoint[1])
            #else:
            #    new_keypoint = (0,0)
            new_keypoint = (0,0,0.0)

        resulting_keypoints.append(new_keypoint)
    
    return resulting_keypoints
    

