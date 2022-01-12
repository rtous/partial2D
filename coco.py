from pycocotools.coco import COCO
import poseUtils
import numpy as np
import cv2

#Check this for converting COCO and OpenPose: https://www.reddit.com/r/computervision/comments/bpx2us/conversion_from_coco_keypoints_to_open_pose/

WIDTH = 128
HEIGHT = 128
SPINESIZE = WIDTH/6
HAVETHRESHOLD = True
THRESHOLD = -1

#Just informative (from, e.g., https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)
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
    #It has no neck, you can add it for drawing or for converting to openpose
} 

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

def addNeck(keypoints):
    #In COCO the neck keypoint is the middle point between shoulders (and located in pos 1)
    l_shoulder_keypoint = keypoints[5]
    r_shoulder_keypoint = keypoints[4]
    if l_shoulder_keypoint[2] > 0 and r_shoulder_keypoint[2] > 0:
        neck_keypoint_x = int((l_shoulder_keypoint[0]+r_shoulder_keypoint[0])/2.)
        neck_keypoint_y = int((l_shoulder_keypoint[1]+r_shoulder_keypoint[1])/2.)
        new_keypoint = (int(neck_keypoint_x), int(neck_keypoint_y), 1)
    else:
        new_keypoint = (0, 0, 0)
    
    keypoints_with_neck = []
    
    neck_pos = 17
    for j in range(0, neck_pos):
        keypoints_with_neck.append(keypoints[j])
    keypoints_with_neck.append(new_keypoint)
    for j in range(neck_pos, len(keypoints)):
        keypoints_with_neck.append(keypoints[j])    
    return keypoints_with_neck


train_annot_path = '/Volumes/ElementsDat/pose/COCO/ruben_structure/person_keypoints_train2017.json'
train_coco = COCO(train_annot_path) # load annotations for training set

# function iterates ofver all ocurrences of a  person and returns relevant data row by row
def get_meta(coco):
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
################################
# Put 25 images into a list
num = 0
annotations = []
for img_id, img_fname, w, h, meta in get_meta(train_coco):
    # iterate over all annotations of an image
    annotations.append([img_id, img_fname, w, h, meta])
    num += 1   
    '''
    for m in meta:
        # m is a dictionary
        keypoints = m['keypoints']  

        
        print(keypoints)
    '''
    if num >= 25:
        break
################################
#Test one image
j = 5
img_id = annotations[j][0]
img_fname = annotations[j][1]
w = annotations[j][2]
h = annotations[j][3] 
m = annotations[j][4][0]
keypointsFlat = m['keypoints']    
keypoints=list(zip(list(map(int, keypointsFlat[0::3])), list(map(int, keypointsFlat[1::3])), list(map(float, keypointsFlat[2::3]))))
#keypointsWithNeck = addNeck(keypoints)
boneSpineIndex = 2
filepath = "/Volumes/ElementsDat/pose/COCO/train2017/"+img_fname
print(filepath)
originalImage = cv2.imread(filepath)
print("w="+str(w)+", h="+str(h))
poseUtils.displayImage(originalImage, w, h)
if poseUtils.hasBone(keypoints, POSE_PAIRS, boneSpineIndex):
    #poseUtils.normalize_pose(keypoints, POSE_PAIRS, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, HAVETHRESHOLD)
    poseUtils.draw_pose(originalImage, keypoints, THRESHOLD, POSE_PAIRS_FOR_DRAWING, POSE_MPI_COLORS_RENDER_GPU, HAVETHRESHOLD)
    #poseUtils.draw_keypoints(blank_image, keypoints)


poseUtils.draw_keypoints(originalImage, keypoints)
poseUtils.displayImage(originalImage, w, h)
################################
#With many
NUM_ROWS = 5
NUM_COLS = 5
images = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
print(len(annotations))
for i in range(NUM_ROWS*NUM_COLS):
    blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)

    #pillo la primera annotació de la imatge si es que n'hi ha
    if len(annotations[i][4])>0:

        m = annotations[i][4][0]
        #print(m)
        keypointsFlat = m['keypoints']    
        keypoints=list(zip(list(map(int, keypointsFlat[0::3])), list(map(int, keypointsFlat[1::3])), list(map(float, keypointsFlat[2::3]))))
        keypoints = addNeck(keypoints)

        boneSpineIndex = 1

        if poseUtils.hasBone(keypoints, POSE_PAIRS, boneSpineIndex):
            poseUtils.normalize_pose(keypoints, POSE_PAIRS, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, HAVETHRESHOLD)
            poseUtils.draw_pose(blank_image, keypoints, THRESHOLD, POSE_PAIRS, POSE_MPI_COLORS_RENDER_GPU, HAVETHRESHOLD)
            poseUtils.draw_keypoints(blank_image, keypoints)

    images[int(i/NUM_COLS)][int(i%NUM_COLS)] = blank_image
    

#total_image = np.hstack(images)

total_image = poseUtils.concat_tile(images)

poseUtils.displayImage(total_image, WIDTH*NUM_COLS, HEIGHT*NUM_ROWS)


