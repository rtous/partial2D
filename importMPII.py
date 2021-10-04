import json
#https://mmpose.readthedocs.io/en/latest/tasks/2d_body_keypoint.html

POSE_MPI_PAIRS_RENDER_GPU = [[0,1],   [1,2],   [2,3],   [3,4],   [1,5],   [5,6],   [6,7],   [1,14],  [14,8],  [8,9],  [9,10],  [14,11], [11,12], [12,13]]
#POSE_MPI_MAP_IDX = [[16,17], [18,19], [20,21], [22,23], [24,25], [26,27], [28,29], [30,31], [32,33], [34,35], [36,37], [38,39], [40,41], [42,43]]    

# joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)

# Opening JSON file
f = open('mpii_train.json',)
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
print(data[0]['joints'])
#person = data['people'][0]

