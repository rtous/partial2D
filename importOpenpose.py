
import json
import cv2
import numpy as np

#https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1644

# From https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/20d8eca4b43fe28cefc02d341476b04c6a6d6ff2/doc/output.md#pose-output-format-body_25

# Openpose code: https://github.com/ArtificialShane/OpenPose/blob/master/include/openpose/pose/poseParameters.hpp (outdated!)

#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/pose/poseParameters.cpp

 
POSE_MPI_BODY_PARTS = {
    0:  "Head",
    1:  "Neck",
    2:  "RShoulder",
    3:  "RElbow",
    4:  "RWrist",
    5:  "LShoulder",
    6:  "LElbow",
    7:  "LWrist",
    8:  "RHip",
    9:  "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "Chest",
    15: "Background"
}

POSE_MPI_COLORS_RENDER_GPU  = [
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
          [0,     0,   255]
]

POSE_BODY_23_COLORS_RENDER_GPU =[
        [255,     0,     0], 
        [255,    55,     0], 
        [255,   110,     0], 
        [255,   165,     0], 
        [255,   215,     0], 
        [255,   235,     0], 
        [255,   255,     0], 
        [255,     0,     0], 
        [175,   255,     0], 
         [85,   255,     0], 
          [0,   255,     0], 
          [0,   255,    85], 
          [0,   255,   170], 
         [25,    25,   128], 
          [0,    85,   255], 
          [0,   170,   255], 
          [0,   212,   255], 
          [0,   255,   255], 
        [255,     0,     0], 
        [255,     0,   255], 
        [238,   130,   238], 
        [138,    43,   226], 
         [138,    43,   226], 
          [138,    43,   226], 
           [138,    43,   226], 
         [75,     0,   130]
]

POSE_MPI_NUMBER_PARTS            = 15 #// Equivalent to size of std::map POSE_MPI_NUMBER_PARTS - 1 (removing background)
#POSE_MPI_MAP_IDX = [16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29, 30,31, 32,33, 34,35, 36,37, 38,39, 40,41, 42,43]

POSE_MPI_PAIRS_RENDER_GPU = [[0,1],   [1,2],   [2,3],   [3,4],   [1,5],   [5,6],   [6,7],   [1,14],  [14,8],  [8,9],  [9,10],  [14,11], [11,12], [12,13]]

#POSE_BODY_23_PAIRS = [[0,1],  [0,4],  [1,2],  [2,3],  [4,5],  [5,6],  [0,7],  [7,8],  [7,13], [8,9],  [9,10],[10,11],[10,12],[13,14],[14,15],[15,16],[15,17], [0,18],[18,19],[18,21],[19,20],[21,22], [1,20], [4,22]]

POSE_BODY_23_PAIRS = [[0,1],  [0,4],  [1,2],  [2,3],  [4,5],  [5,6],  [0,7],  [7,8],  [7,13], [8,9],  [9,10],[10,11],[10,12],[13,14],[14,15],[15,16],[15,17], [0,18],[18,19],[18,21],[19,20],[21,22], [1,20]]

POSE_BODY_PART_PAIRS_BODY_25 = [[1,8],   [1,2],   [1,5],   [2,3],   [3,4],   [5,6],   [6,7],   [8,9],   [9,10],  [10,11], [8,12],  [12,13], [13,14],  [1,0],   [0,15], [15,17],  [0,16], [16,18],   [2,17],  [5,18],   [14,19],[19,20],[14,21], [11,22],[22,23],[11,24]]


#BONES = [[0,1],   [1,2],   [2,3],   [3,4],   [1,5],   [5,6],   [6,7],   [1,14]]

#def keypointsToBones(keypoints):
#     for idx, k in enumerate(keypoints):
     

def draw_pose(img, keypoints, threshold=0.2):
    '''
    for idx, k in enumerate(keypoints):
        x = int(k[0])
        y = int(k[1])
        c = float(k[2])
        img = cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
        if idx > 0:
            px = int(previous[0])
            py = int(previous[1])
            pc = float(previous[2])
            if pc != 0 and c != 0:
                img = cv2.line(img, (px, py), (x, y), (0, idx * 10, 255), 2)
        previous = k
    '''   
          
    for boneNumer, bone in enumerate(POSE_BODY_PART_PAIRS_BODY_25):
        print("Part: "+str(bone[0])+","+str(bone[1]))
        keypoint1 = keypoints[bone[0]]
        keypoint2 = keypoints[bone[1]]
        color = POSE_BODY_23_COLORS_RENDER_GPU[boneNumer]
        
        print(keypoint1)
        
        if keypoint1[2] > threshold and keypoint2[2] > threshold:
            img = cv2.line(img, (keypoint1[0], keypoint1[1]), (keypoint2[0], keypoint2[1]), (color[0], color[1], color[2]), 2)
        
        
        

 
# Opening JSON file
f = open('001_keypoints.json',)
 
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
print(str(len(POSE_BODY_23_PAIRS))+" parts found")
print(str(len(POSE_BODY_23_COLORS_RENDER_GPU))+" colors found")

for idx, k in enumerate(keypoints):
    x = k[0]
    y = k[1]
    c = k[2]
    print (str(x) +',' + str(y) + '(' + str(c) + ')')

blank_image = np.zeros((500,500,3), np.uint8)

draw_pose(blank_image, keypoints, threshold=0.2)

#cv2.drawKeypoints(blank_image, kps5, img_brisk, kps5, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
cv2.imshow('Test', blank_image)
cv2.resizeWindow('Test', 600,600)

k = cv2.waitKey(0) & 0xFF

if k == 27:         # wait for ESC key to exit
  cv2.destroyAllWindows()
else:
  print("Exit")


 
# Closing file
f.close()