
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
THRESHOLD = 0 #disabled #0.05 #0.05
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

def normalizedKeypoints2json(keypoints, path):
    '''
    {"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": [560.046, 254.438, 0.956364, 579.055, 287.017, 0.945341, 554.58, 289.731, 0.933184, 527.202, 300.732, 0.802688, 510.957, 311.574, 0.878938, 601.005, 284.288, 0.959404, 633.701, 322.576, 0.925591, 655.588, 366.202, 0.923742, 581.996, 382.658, 0.924048, 568.311, 382.697, 0.878411, 579.271, 459.06, 0.944777, 590.088, 530.022, 0.935949, 600.911, 382.618, 0.933565, 598.347, 461.771, 0.951028, 603.761, 538.2, 0.90781, 557.229, 251.571, 0.97114, 568.165, 248.972, 0.964149, 554.653, 254.308, 0.0591213, 581.807, 251.6, 0.963746, 587.377, 551.834, 0.660232, 598.261, 554.523, 0.730152, 606.492, 546.402, 0.813981, 565.52, 543.672, 0.917552, 562.853, 538.168, 0.901656, 598.199, 538.176, 0.625304], "face_keypoints_2d": [567.755, 285.861, 0.0264805, 567.755, 285.861, 0.0011107, 560.488, 263.614, 0.0405039, 562.713, 255.754, 0.0273063, 567.31, 287.493, 0.00312189, 554.852, 256.94, 0.0260331, 582.883, 270.14, 0.0289196, 554.555, 256.792, 0.00992974, 555.89, 256.347, 0.0199999, 591.486, 269.399, 0.00610582, 560.784, 250.118, 0.0144156, 560.784, 267.915, 0.0280541, 577.544, 270.585, 0.0334872, 579.917, 256.94, 0.0469211, 577.989, 262.279, 0.0713802, 579.027, 251.304, 0.0441189, 576.951, 245.075, 0.0497181, 552.627, 256.644, 0.00119348, 552.479, 256.495, 0.0259865, 552.924, 255.754, 0.0452728, 553.22, 255.605, 0.0587203, 553.072, 255.457, 0.00173667, 558.56, 258.275, 0.00131636, 556.78, 257.088, 0.00675738, 558.263, 258.423, 0.0748387, 558.56, 257.237, 0.0774293, 559.153, 256.347, 0.0351459, 557.225, 255.16, 0.00075239, 558.263, 253.826, 0.00742681, 555.89, 255.605, 0.0161634, 556.038, 258.127, 0.021296, 555.89, 259.313, 0.0286869, 556.038, 258.868, 0.0281499, 558.411, 259.61, 0.072706, 559.45, 259.313, 0.101319, 560.784, 258.275, 0.138944, 570.128, 259.61, 0.0291072, 568.793, 259.906, 0.0109359, 553.517, 255.754, 0.00075579, 553.517, 255.902, 0.000373998, 565.679, 260.945, 0.00962272, 570.128, 259.758, 0.00728712, 559.598, 257.682, 0.0167088, 559.598, 257.682, 0.0724896, 559.895, 258.275, 0.0390386, 561.823, 257.533, 0.0216627, 560.191, 258.127, 0.0437787, 560.043, 257.83, 0.0655467, 557.225, 249.524, 0.013698, 556.632, 251.304, 0.0161038, 559.598, 263.021, 0.0183623, 560.191, 262.428, 0.0156363, 560.933, 260.351, 0.0302365, 561.674, 260.203, 0.0472756, 562.268, 262.428, 0.0276113, 562.713, 262.428, 0.0259921, 562.119, 262.724, 0.0173563, 555.445, 252.342, 0.0152972, 556.038, 250.859, 0.0168393, 556.928, 250.414, 0.0085928, 556.78, 249.673, 0.0170601, 558.708, 262.279, 0.031341, 560.043, 262.724, 0.0200892, 562.119, 261.835, 0.00535929, 560.784, 257.533, 0.0407231, 562.564, 262.279, 0.0078524, 560.784, 262.131, 0.0336926, 560.339, 262.279, 0.0247457, 553.665, 255.605, 0.00161107, 560.488, 257.83, 0.058714], "hand_left_keypoints_2d": [654.653, 370.453, 0.388057, 651.072, 376.222, 0.512436, 650.276, 384.578, 0.760358, 653.062, 390.347, 0.801366, 654.852, 395.321, 0.877627, 658.234, 384.976, 0.649731, 659.627, 393.133, 0.78041, 659.826, 397.111, 0.87892, 659.229, 400.096, 0.853599, 661.815, 384.379, 0.580774, 663.606, 391.939, 0.69692, 662.014, 395.52, 0.559785, 660.423, 397.907, 0.428586, 664.004, 383.185, 0.620956, 665.595, 388.756, 0.60064, 664.402, 392.337, 0.57832, 661.616, 394.326, 0.431797, 665.595, 381.992, 0.617871, 666.59, 386.567, 0.63573, 665.595, 389.154, 0.473403, 664.004, 390.546, 0.434321], "hand_right_keypoints_2d": [512.415, 331.496, 0.153315, 516.853, 305.302, 0.0561528, 516.42, 303.786, 0.0254495, 503.864, 307.467, 0.0352869, 502.89, 306.384, 0.152461, 503.864, 308.332, 0.0299997, 513.389, 317.208, 0.012805, 503.648, 307.791, 0.0160843, 502.782, 306.817, 0.0238978, 504.297, 310.93, 0.026109, 506.462, 316.667, 0.0143305, 503.215, 311.471, 0.0233719, 502.89, 306.925, 0.0155019, 508.194, 310.606, 0.0353555, 513.065, 312.446, 0.0364679, 507.328, 319.373, 0.0212762, 503.215, 322.079, 0.018511, 503.864, 323.053, 0.0286689, 513.065, 316.775, 0.0485211, 514.147, 317.316, 0.0551642, 503.107, 322.079, 0.0277985], "pose_keypoints_3d": [], "face_keypoints_3d": [], "hand_left_keypoints_3d": [], "hand_right_keypoints_3d": []}]}
    '''

    data = {"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": [], "pose_keypoints_3d": [], "face_keypoints_3d": [], "face_keypoints_2d": [], "hand_left_keypoints_3d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_3d": [], "hand_right_keypoints_2d": []}]}
    
    #keypointsWithHandsAndFace =addHandsAndFace(keypoints)

    keypointsWithConfidence = addConfidenceValue(keypoints)

    keypointsFlat = np.array(keypointsWithConfidence).flatten()

    data['people'][0]['pose_keypoints_2d'] = keypointsFlat.tolist()

    data['people'][0]['face_keypoints_2d'] = dummyKeypoints(70).tolist()

    data['people'][0]['hand_right_keypoints_2d'] = dummyKeypoints(21).tolist()

    data['people'][0]['hand_left_keypoints_2d'] = dummyKeypoints(21).tolist()

    outfile = open(path, 'w')

    json.dump(data, outfile)
    
    outfile.close()

'''
def addHandsAndFace(keypoints):
    
    for i in range(137-25):
        new_keypoint = (0, 0)
        keypoints.append(new_keypoint)
    return keypoints
'''

def dummyKeypoints(howMany):
    dummy = []
    for i in range(howMany):
        new_keypoint = (0, 0, 0.0)
        dummy.append(new_keypoint)
    dummy = np.array(dummy).flatten()
    return dummy

def addConfidenceValue(keypoints):

    for i, k in enumerate(keypoints):
        new_keypoint = (k[0], k[1], 1.0)
        keypoints[i] = new_keypoint
    return keypoints

def normalize(keypoints):

    boneSpineIndex = findPart('Nose')

    poseUtils.normalize_pose(keypoints, POSE_BODY_25_PAIRS_RENDER_GP, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, False)

    return keypoints

def removeConfidence(keypoints):
    #If confidence below theshold (0.1) the keypoint will be 0,0
    for i, k in enumerate(keypoints):
        
        if k[2] > THRESHOLD: 
            new_keypoint = (k[0], k[1])
        else:
            new_keypoint = (0.0, 0.0)
        #new_keypoint = (k[0], k[1])
        keypoints[i] = new_keypoint
    return keypoints

