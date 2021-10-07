
import json
import cv2
import numpy as np
import math

def draw_part(img, keypoint1, keypoint2, color, keypoint_index_pairs):
	img = cv2.line(img, (keypoint1[0], keypoint1[1]), (keypoint2[0], keypoint2[1]), (color[0], color[1], color[2]), 2)
        
def draw_pose(img, keypoints, threshold, keypoint_index_pairs, colors, haveThreshold):        
    for boneNumer, bone in enumerate(keypoint_index_pairs):
        print("Part: "+str(bone[0])+","+str(bone[1]))
        keypoint1 = keypoints[bone[0]]
        keypoint2 = keypoints[bone[1]]
        color = colors[boneNumer]
        
        print(keypoint1)
        
        if haveThreshold:
            if keypoint1[2] > threshold and keypoint2[2] > threshold:
                draw_part(img, keypoint1, keypoint2, color, keypoint_index_pairs)
        else:
                draw_part(img, keypoint1, keypoint2, color, keypoint_index_pairs)

def draw_keypoints(img, keypoints):        
    for k in keypoints:
        cv2.circle(img,(k[0],k[1]), 1, (0,255,0), thickness=5, lineType=8, shift=0)

def normalize_pose(keypoints, keypoint_index_pairs, spinesize, width, height, boneSpineIndex, haveThreshold):
	
	keypoint1 = keypoints[keypoint_index_pairs[boneSpineIndex][0]]
	keypoint2 = keypoints[keypoint_index_pairs[boneSpineIndex][1]]

	if keypoint1[0]!=0 and keypoint1[1]!=0 and keypoint2[0]!=0 and keypoint2[1]!=0:

		x_distance = keypoint1[0]-keypoint2[0]
		y_distance = keypoint1[1]-keypoint2[1]

		#Compute the length of the spine
		magnitudeSpine = math.sqrt(pow(x_distance, 2)+pow(y_distance, 2))

		#Compute scale factor to obtain the desired spine size
		scaleFactor = magnitudeSpine/spinesize

		#Scale: dividing any keypoint by the scale factor
		for i, k in enumerate(keypoints):
			if haveThreshold:
				new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor), k[2]) 
			else: 
				new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor)) 
			keypoints[i] = new_keypoint

		#Center to the top of the spine
		keypoint1 = keypoints[keypoint_index_pairs[boneSpineIndex][1]]
		x_displacement = keypoint1[0] - int(width/2)
		y_displacement = keypoint1[1] - int(height/2)
		for i, k in enumerate(keypoints):
			if haveThreshold:
				new_keypoint = (int(k[0]-x_displacement), int(k[1]-y_displacement), k[2]) 
			else:
				new_keypoint = (int(k[0]-x_displacement), int(k[1]-y_displacement))
			keypoints[i] = new_keypoint

	return keypoints

def displayImage(image, width, height):
	cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
	cv2.imshow('Test', image)
	cv2.resizeWindow('Test', width, height)

	k = cv2.waitKey(0) & 0xFF

	if k == 27:         # wait for ESC key to exit
	  cv2.destroyAllWindows()
	else:
	  print("Exit")

def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


