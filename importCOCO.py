import json
import numpy as np
import poseUtils

WIDTH = 128
HEIGHT = 128
SPINESIZE = WIDTH/6
HAVETHRESHOLD = True
THRESHOLD = 1.0

#https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1273

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]
#For COCO, part 1 is the neck position calculated by the mean of the two shoulders
#['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'].
# 0 -nose'
# 1 - neck (calculated) 
# 2 -left_eye
# 3 - right_eye
# 4 - left_ear
# 5 - right_ear
# 6 - left_shoulder
# 7 - right_shoulder
# 8 - left_elbow
# 9 - right_elbow
# 10 - left_wrist
# 11 - right_wrist
# 12 - left_hip
# 13 - right_hip 
# 14 - left_knee
# 15 - right_knee
# 16 - left_ankle
# 17 - right_ankle
#
#
#format= 17x3


#POSE_COCO_PAIRS_RENDER_GPU \
#        1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17
    


keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']


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


#POSE_MPI_MAP_IDX = [[16,17], [18,19], [20,21], [22,23], [24,25], [26,27], [28,29], [30,31], [32,33], [34,35], [36,37], [38,39], [40,41], [42,43]]    

# joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)

# Opening JSON file
f = open('COCO_person_keypoints_train2017.json',)
 
# returns JSON object as
# a dictionary
data = json.load(f)

#num_keypoints
#for i in range(10):
#  print(data['annotations'][i]['num_keypoints'])
#  print(data['annotations'][i]['keypoints'])
 
NUM_ROWS = 5
NUM_COLS = 5
images = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
for i in range(NUM_ROWS*NUM_COLS):
	keypointsFlat = data['annotations'][i]['keypoints']
  #keypoints = list(zip(list(map(int, keypointsFlat[0::3])), list(map(int, keypointsFlat[1::3])), list(map(float, keypointsFlat[2::3]))))
	keypoints=list(zip(list(map(int, keypointsFlat[0::3])), list(map(int, keypointsFlat[1::3])), list(map(float, keypointsFlat[2::3]))))
  #print(data[16]['joints'])
	#person = data['people'][0]

	blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)

	#To integers
	for j, k in enumerate(keypoints):
		print(k)
		new_keypoint = (int(k[0]), int(k[1]), int(k[2]))
		keypoints[j] = new_keypoint

	#In COCO the neck keypoint is the middle point between shoulders (and located in pos 1)
	l_shoulder_keypoint = keypoints[5]
	r_shoulder_keypoint = keypoints[6]
	if l_shoulder_keypoint[2] > 0 and r_shoulder_keypoint[2] > 0:
		neck_keypoint_x = int((l_shoulder_keypoint[0]+r_shoulder_keypoint[0])/2.)
		neck_keypoint_y = int((l_shoulder_keypoint[1]+r_shoulder_keypoint[1])/2.)
		new_keypoint = (int(neck_keypoint_x), int(neck_keypoint_y), 1)
	else:
		new_keypoint = (0, 0, 0)
	
	keypoints_with_neck = []
	keypoints_with_neck.append(keypoints[0])
	keypoints_with_neck.append(new_keypoint)
	for j in range(1, len(keypoints)):
		keypoints_with_neck.append(keypoints[j])

	print("keypoints_with_neck:")
	print(keypoints_with_neck)

	boneSpineIndex = 1

	poseUtils.normalize_pose(keypoints_with_neck, POSE_PAIRS, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, HAVETHRESHOLD)
	poseUtils.draw_pose(blank_image, keypoints_with_neck, THRESHOLD, POSE_PAIRS, POSE_MPI_COLORS_RENDER_GPU, HAVETHRESHOLD)
	poseUtils.draw_keypoints(blank_image, keypoints_with_neck)

	images[int(i/NUM_COLS)][int(i%NUM_COLS)] = blank_image
	

#total_image = np.hstack(images)

total_image = poseUtils.concat_tile(images)

poseUtils.displayImage(total_image, WIDTH*NUM_COLS, HEIGHT*NUM_ROWS)

# Closing file
f.close()

