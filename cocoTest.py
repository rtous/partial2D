from pycocotools.coco import COCO
import poseUtils
import numpy as np
import cv2
import cocoUtils

#Check this for converting COCO and OpenPose: https://www.reddit.com/r/computervision/comments/bpx2us/conversion_from_coco_keypoints_to_open_pose/

WIDTH = 128
HEIGHT = 128
SPINESIZE = WIDTH/6
HAVETHRESHOLD = True
THRESHOLD = -1

# Put 25 images into a list
num = 0
annotations = []
for img_id, img_fname, w, h, meta in cocoUtils.getCOCOAnnotations():
    # iterate over all annotations of an image
    annotations.append([img_id, img_fname, w, h, meta])
    num += 1   
    if num >= 25:
        break

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
if poseUtils.hasBone(keypoints, cocoUtils.POSE_PAIRS_FOR_DRAWING, boneSpineIndex):
    #poseUtils.normalize_pose(keypoints, POSE_PAIRS, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, HAVETHRESHOLD)
    poseUtils.draw_pose(originalImage, keypoints, THRESHOLD, cocoUtils.POSE_PAIRS_FOR_DRAWING, cocoUtils.POSE_MPI_COLORS_RENDER_GPU, HAVETHRESHOLD)
    #poseUtils.draw_keypoints(blank_image, keypoints)


poseUtils.draw_keypoints(originalImage, keypoints)
poseUtils.displayImage(originalImage, w, h)

#test with many images
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
        #keypoints = addNeck(keypoints)

        boneSpineIndex = 1

        if poseUtils.hasBone(keypoints, cocoUtils.POSE_PAIRS_FOR_DRAWING, boneSpineIndex):
            print("Drawn ('has') reference bone): ")
            print(keypoints)
            poseUtils.normalize_pose(keypoints, cocoUtils.POSE_PAIRS_FOR_DRAWING, SPINESIZE, WIDTH, HEIGHT, boneSpineIndex, HAVETHRESHOLD)
            poseUtils.draw_pose(blank_image, keypoints, THRESHOLD, cocoUtils.POSE_PAIRS_FOR_DRAWING, cocoUtils.POSE_MPI_COLORS_RENDER_GPU, HAVETHRESHOLD)
            #poseUtils.draw_keypoints(blank_image, keypoints)
        else:
            print("Not drawn, ('does not have') reference bone): ")
            print(keypoints)

    images[int(i/NUM_COLS)][int(i%NUM_COLS)] = blank_image
    
total_image = poseUtils.concat_tile(images)

poseUtils.displayImage(total_image, WIDTH*NUM_COLS, HEIGHT*NUM_ROWS)


