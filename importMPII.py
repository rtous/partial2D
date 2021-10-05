import json
import numpy as np
import poseUtils

WIDTH = 128
HEIGHT = 128
SPINESIZE = WIDTH/4
HAVETHRESHOLD = False


#https://mmpose.readthedocs.io/en/latest/tasks/2d_body_keypoint.html
#joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)

#POSE_MPI_PAIRS_RENDER_GPU = [[0,1],   [1,2],   [2,3],   [3,4],   [1,5],   [5,6],   [6,7],   [1,14],  [14,8],  [8,9],  [9,10],  [14,11], [11,12], [12,13]]

POSE_MPI_PAIRS_RENDER_GPU = [[9,8],   [8,7],  #[7,6],  #7 - thorax, 8 - upper neck, 9 - head top, 6 - pelvis
                             [12, 2], [13, 3],
                             [6,2],   [2,1],  [1,0],   #0 - r ankle, 1 - r knee, 2 - r hip
                             [6,3],   [3,4],  [4,5],   #3 - l hip, 4 - l knee, 5 - l ankle, 
                             [7,12],  [12,11], [11,10],#10 - r wrist, 11 - r elbow, 12 - r shoulder, 
                             [7,13],  [13,14], [14,15]] #13 - l shoulder, 14 - l elbow, 15 - l wrist)

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
          [0,     0,   255]
 ]


#POSE_MPI_MAP_IDX = [[16,17], [18,19], [20,21], [22,23], [24,25], [26,27], [28,29], [30,31], [32,33], [34,35], [36,37], [38,39], [40,41], [42,43]]    

# joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)

# Opening JSON file
f = open('mpii_train.json',)
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
NUM_ROWS = 5
NUM_COLS = 5
images = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
for i in range(NUM_ROWS*NUM_COLS):
	keypoints = data[i]['joints']
	#print(data[16]['joints'])
	#person = data['people'][0]

	blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)

	boneSpineIndex = 2

	#To integers
	for k in keypoints:
			k[0] = int(k[0])
			k[1] = int(k[1])

	poseUtils.normalize_pose(keypoints, POSE_MPI_PAIRS_RENDER_GPU, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, HAVETHRESHOLD)
	poseUtils.draw_pose(blank_image, keypoints, 0, POSE_MPI_PAIRS_RENDER_GPU, POSE_MPI_COLORS_RENDER_GPU, HAVETHRESHOLD)
	poseUtils.draw_keypoints(blank_image, keypoints)

	images[int(i/NUM_COLS)][int(i%NUM_COLS)] = blank_image
	

#total_image = np.hstack(images)

total_image = poseUtils.concat_tile(images)

poseUtils.displayImage(total_image, WIDTH*NUM_COLS, HEIGHT*NUM_ROWS)

# Closing file
f.close()

