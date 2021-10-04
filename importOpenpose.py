
import json
import cv2
import numpy as np
import math

WIDTH = 64
HEIGHT = 64
SPINESIZE = WIDTH/4

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

def draw_part(img, keypoint1, keypoint2, color):
	img = cv2.line(img, (keypoint1[0], keypoint1[1]), (keypoint2[0], keypoint2[1]), (color[0], color[1], color[2]), 2)
        
def draw_pose(img, keypoints, threshold=0.1):        
    for boneNumer, bone in enumerate(POSE_BODY_25_PAIRS_RENDER_GP):
        print("Part: "+str(bone[0])+","+str(bone[1]))
        keypoint1 = keypoints[bone[0]]
        keypoint2 = keypoints[bone[1]]
        color = POSE_BODY_25_COLORS_RENDER_GPU[boneNumer]
        
        print(keypoint1)
        
        if keypoint1[2] > threshold and keypoint2[2] > threshold:
            #img = cv2.line(img, (keypoint1[0], keypoint1[1]), (keypoint2[0], keypoint2[1]), (color[0], color[1], color[2]), 2)
        	draw_part(img, keypoint1, keypoint2, color)

'''
def scale_bone(boneIndex, scaleFactor):
	#scaleFactor = 0.5
	keypoint1 = keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneIndex][0]]
	keypoint2 = keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneIndex][1]]
	x_distance = keypoint1[0]-keypoint2[0]
	y_distance = keypoint1[1]-keypoint2[1]

	new_keypoint1 = (int(keypoint1[0]*scaleFactor), int(keypoint1[1]*scaleFactor), keypoint1[2])	
	new_keypoint2 = (int(keypoint2[0]*scaleFactor), int(keypoint2[1]*scaleFactor), keypoint2[2])	
	
	#new_keypoint2 =  (int(new_keypoint1[0]+x_distance/scaleFactor), int(new_keypoint1[1]+y_distance/scaleFactor), keypoint2[2])
	
	print("new_keypoint1=",new_keypoint1)
	print("new_keypoint2=",new_keypoint2)

	keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneIndex][0]]= new_keypoint1
	keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneIndex][1]]= new_keypoint2
'''
#def scale_keypoint(keypointIndex):




def normalize_pose(keypoints):
	#keypoints_norm = []

	boneSpineIndex = findPart('Nose')
	keypoint1 = keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneSpineIndex][0]]
	keypoint2 = keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneSpineIndex][1]]
	x_distance = keypoint1[0]-keypoint2[0]
	y_distance = keypoint1[1]-keypoint2[1]
	#Normalize: divide each component by its magnitude
	magnitudeSpine = math.sqrt(pow(x_distance, 2)+pow(y_distance, 2))
	magnitudeSpine = magnitudeSpine/SPINESIZE
	print("magnitudeSpine=",magnitudeSpine)

	#x_displacement = keypoint1[0] - int(WIDTH/2)
	#y_displacement = keypoint1[1] - int(HEIGHT/2)

	'''
	new_keypoint1 = (int(WIDTH/2), int(HEIGHT/2), keypoint1[2]) 
	new_keypoint2 =  (int(new_keypoint1[0]+x_distance/magnitudeSpine), int(new_keypoint1[1]+y_distance/magnitudeSpine), keypoint2[2])
	print("new_keypoint1=",new_keypoint1)
	print("new_keypoint2=",new_keypoint2)

	keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneSpineIndex][0]]= new_keypoint1
	keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneSpineIndex][1]]= new_keypoint2
	'''

	#for boneNumer, bone in enumerate(POSE_BODY_25_PAIRS_RENDER_GP):
	#	scale_bone(boneNumer, magnitudeSpine)

	#SCALE
	for i, k in enumerate(keypoints):
		new_keypoint = (int(k[0]/magnitudeSpine), int(k[1]/magnitudeSpine), k[2]) 
		keypoints[i] = new_keypoint

	#CENTER SPINE
	boneSpineIndex = findPart('Nose')
	keypoint1 = keypoints[POSE_BODY_25_PAIRS_RENDER_GP[boneSpineIndex][1]]
	x_displacement = keypoint1[0] - int(WIDTH/2)
	y_displacement = keypoint1[1] - int(HEIGHT/2)
	for i, k in enumerate(keypoints):
		new_keypoint = (int(k[0]-x_displacement), int(k[1]-y_displacement), k[2]) 
		keypoints[i] = new_keypoint

	return keypoints


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

#print(keypoints)
print(str(len(keypoints))+" keypoints found")
print(str(len(POSE_BODY_25_PAIRS_RENDER_GP))+" parts found")

'''
for idx, k in enumerate(keypoints):
    x = k[0]
    y = k[1]
    c = k[2]
    print (str(x) +',' + str(y) + '(' + str(c) + ')')
'''

blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)



normalize_pose(keypoints)

draw_pose(blank_image, keypoints, threshold=0.2)

'''
neckIndex = findPart("Nose")
#print("neckIndex="+neckIndex)
neckJoints = POSE_BODY_25_PAIRS_RENDER_GP[neckIndex]
draw_part(blank_image, keypoints[neckJoints[0]], keypoints[neckJoints[1]], POSE_BODY_25_COLORS_RENDER_GPU[neckIndex])
'''

'''
neckIndex = findPart("RShoulder")
#print("neckIndex="+neckIndex)
neckJoints = POSE_BODY_25_PAIRS_RENDER_GP[neckIndex]
draw_part(blank_image, keypoints[neckJoints[0]], keypoints[neckJoints[1]], POSE_BODY_25_COLORS_RENDER_GPU[neckIndex])

neckIndex = findPart("RElbow")
#print("neckIndex="+neckIndex)
neckJoints = POSE_BODY_25_PAIRS_RENDER_GP[neckIndex]
draw_part(blank_image, keypoints[neckJoints[0]], keypoints[neckJoints[1]], POSE_BODY_25_COLORS_RENDER_GPU[neckIndex])
'''


#cv2.drawKeypoints(blank_image, kps5, img_brisk, kps5, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
cv2.imshow('Test', blank_image)
cv2.resizeWindow('Test', WIDTH,HEIGHT)

k = cv2.waitKey(0) & 0xFF

if k == 27:         # wait for ESC key to exit
  cv2.destroyAllWindows()
else:
  print("Exit")

# Closing file
f.close()




