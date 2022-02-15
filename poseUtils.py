
import json
import cv2
import numpy as np
import math
import sys

def keypointsListFlatten(keypoints):
	keypointsNP = np.vstack(keypoints)
	keypointsNPflat = keypointsNP.flatten()
	return keypointsNPflat

def geJointIndexFromName(jointName, jointNamesDict):
    idx = list(jointNamesDict.keys())[list(jointNamesDict.values()).index(jointName)]
    return idx 

def computeMiddleJoint(keypoints, joint1name, joint2name, DICT):
    #Middle point between shoulders
    keypoint1 = keypoints[geJointIndexFromName(joint1name, DICT)]
    keypoint2 = keypoints[geJointIndexFromName(joint2name, DICT)]
    if keypoint1[2] > 0 and keypoint2[2] > 0:
        middle_keypoint_x = int((keypoint1[0]+keypoint2[0])/2.)
        middle_keypoint_y = int((keypoint1[1]+keypoint2[1])/2.)
        new_keypoint = (int(middle_keypoint_x), int(middle_keypoint_y), 1)
    else:
        new_keypoint = (0, 0, 0)
    return new_keypoint

def draw_part(img, keypoint1, keypoint2, color, keypoint_index_pairs, thickness):
	#img = cv2.line(img, (keypoint1[0], keypoint1[1]), (keypoint2[0], keypoint2[1]), (color[0], color[1], color[2]), 2)
    img = cv2.line(img, (int(keypoint1[0]), int(keypoint1[1])), (int(keypoint2[0]), int(keypoint2[1])), (color[0], color[1], color[2]), thickness)
    
def draw_pose(img, keypoints, threshold, keypoint_index_pairs, colors, haveThreshold, thickness=2):        
	for boneNumer, bone in enumerate(keypoint_index_pairs):
		#print("Part: "+str(bone[0])+","+str(bone[1]))
		#print(bone)
		keypoint1 = keypoints[bone[0]]
		keypoint2 = keypoints[bone[1]]
		color = colors[boneNumer]
		if haveThreshold:
			if keypoint1[2] > threshold and keypoint2[2] > threshold:
				if keypoint1[0] > 0 and keypoint1[1] > 0 and keypoint2[0] > 0 and keypoint2[1] > 0:
					draw_part(img, keypoint1, keypoint2, color, keypoint_index_pairs, thickness)
		else:
			if keypoint1[0] > 0 and keypoint1[1] > 0 and keypoint2[0] > 0 and keypoint2[1] > 0:
				draw_part(img, keypoint1, keypoint2, color, keypoint_index_pairs, thickness)

def draw_keypoints(img, keypoints):   
	for i, k in enumerate(keypoints):
		cv2.circle(img,(k[0],k[1]), 1, (0,255,0), thickness=4, lineType=8, shift=0)
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (k[0],k[1])
		fontScale              = 0.5
		fontColor              = (255,255,255)#(124,252,0)
		thickness              = 1
		lineType               = 2
		cv2.putText(img,str(i), 
		    bottomLeftCornerOfText, 
		    font, 
		    fontScale,
		    fontColor,
		    thickness,
		    lineType)


def hasBone(keypoints, keypoint_index_pairs, index):
    keypoint1 = keypoints[keypoint_index_pairs[index][0]]
    keypoint2 = keypoints[keypoint_index_pairs[index][1]]

    if keypoint1[0]!=0 and keypoint1[1]!=0 and keypoint2[0]!=0 and keypoint2[1]!=0:
        return True
    else:
        return False

def normalize_pose(keypoints, keypoint_index_pairs, spinesize, width, height, boneSpineIndex, keepThreshold):
	
	normalized_keypoints = keypoints.copy()

	keypoint1 = normalized_keypoints[keypoint_index_pairs[boneSpineIndex][0]]
	keypoint2 = normalized_keypoints[keypoint_index_pairs[boneSpineIndex][1]]

	scaleFactor = -1
	x_displacement = -1
	y_displacement = -1

	if keypoint1[0]!=0 and keypoint1[1]!=0 and keypoint2[0]!=0 and keypoint2[1]!=0:

		x_distance = keypoint1[0]-keypoint2[0]
		y_distance = keypoint1[1]-keypoint2[1]

		#Compute the length of the spine
		magnitudeSpine = math.sqrt(pow(x_distance, 2)+pow(y_distance, 2))

		#Compute scale factor to obtain the desired spine size
		scaleFactor = magnitudeSpine/spinesize

		#Scale: dividing any keypoint by the scale factor
		for i, k in enumerate(normalized_keypoints):
			if keepThreshold:
				#new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor), k[2]) 
				new_keypoint = (k[0]/scaleFactor, k[1]/scaleFactor, k[2]) 
			else: 
				#new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor)) 
				new_keypoint = (k[0]/scaleFactor, k[1]/scaleFactor) 
			normalized_keypoints[i] = new_keypoint

		#Center to the top of the spine
		keypoint1 = normalized_keypoints[keypoint_index_pairs[boneSpineIndex][1]]
		x_displacement = keypoint1[0] - width/2
		y_displacement = keypoint1[1] - height/2
		for i, k in enumerate(normalized_keypoints):
			if keepThreshold:
				#new_keypoint = (int(k[0]-x_displacement), int(k[1]-y_displacement), k[2]) 
				new_keypoint = (k[0]-x_displacement, k[1]-y_displacement, k[2]) 
			else:
				#new_keypoint = (int(k[0]-x_displacement), int(k[1]-y_displacement))
				new_keypoint = (k[0]-x_displacement, k[1]-y_displacement)
			normalized_keypoints[i] = new_keypoint

	else:
		raise Exception('Reference bone keypoints are zero :-(')


	return normalized_keypoints, scaleFactor, x_displacement, y_displacement

'''
def normalize_pose(keypoints, keypoint_index_pairs, spinesize, width, height, boneSpineIndex, keepThreshold):
	
	keypoint1 = keypoints[keypoint_index_pairs[boneSpineIndex][0]]
	keypoint2 = keypoints[keypoint_index_pairs[boneSpineIndex][1]]

	scaleFactor = -1
	x_displacement = -1
	y_displacement = -1

	if keypoint1[0]!=0 and keypoint1[1]!=0 and keypoint2[0]!=0 and keypoint2[1]!=0:

		x_distance = keypoint1[0]-keypoint2[0]
		y_distance = keypoint1[1]-keypoint2[1]

		#Compute the length of the spine
		magnitudeSpine = math.sqrt(pow(x_distance, 2)+pow(y_distance, 2))

		#Compute scale factor to obtain the desired spine size
		scaleFactor = magnitudeSpine/spinesize

		#Scale: dividing any keypoint by the scale factor
		for i, k in enumerate(keypoints):
			if keepThreshold:
				#new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor), k[2]) 
				new_keypoint = (k[0]/scaleFactor, k[1]/scaleFactor, k[2]) 
			else: 
				#new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor)) 
				new_keypoint = (k[0]/scaleFactor, k[1]/scaleFactor) 
			keypoints[i] = new_keypoint

		#Center to the top of the spine
		keypoint1 = keypoints[keypoint_index_pairs[boneSpineIndex][1]]
		x_displacement = keypoint1[0] - width/2
		y_displacement = keypoint1[1] - height/2
		for i, k in enumerate(keypoints):
			if keepThreshold:
				#new_keypoint = (int(k[0]-x_displacement), int(k[1]-y_displacement), k[2]) 
				new_keypoint = (k[0]-x_displacement, k[1]-y_displacement, k[2]) 
			else:
				#new_keypoint = (int(k[0]-x_displacement), int(k[1]-y_displacement))
				new_keypoint = (k[0]-x_displacement, k[1]-y_displacement)
			keypoints[i] = new_keypoint

	else:
		raise Exception('Reference bone keypoints are zero :-(')


	return scaleFactor, x_displacement, y_displacement
'''

def denormalize_pose(keypoints, scaleFactor, x_displacement, y_displacement, keepThreshold):
	newKeypoints = [] 
	for i, k in enumerate(keypoints):
		if scaleFactor != -1:
			if keepThreshold:
				#new_keypoint = (int((k[0]+x_displacement)*scaleFactor), int((k[1]+y_displacement)*scaleFactor), k[2]) 
				new_keypoint = ((k[0]+x_displacement)*scaleFactor, (k[1]+y_displacement)*scaleFactor, k[2]) 
			else: 
				#new_keypoint = (int((k[0]+x_displacement)*scaleFactor), int((k[1]+y_displacement)*scaleFactor))	
				new_keypoint = ((k[0]+x_displacement)*scaleFactor, (k[1]+y_displacement)*scaleFactor)	
		
			#keypoints[i] = new_keypoint
			newKeypoints.append(new_keypoint)
	return newKeypoints
'''
def denormalize_pose(keypoints, scaleFactor, x_displacement, y_displacement, keepThreshold):
	#newKeypoints = [] 
	for i, k in enumerate(keypoints):
		if scaleFactor != -1:
			if keepThreshold:
				#new_keypoint = (int((k[0]+x_displacement)*scaleFactor), int((k[1]+y_displacement)*scaleFactor), k[2]) 
				new_keypoint = ((k[0]+x_displacement)*scaleFactor, (k[1]+y_displacement)*scaleFactor, k[2]) 
			else: 
				#new_keypoint = (int((k[0]+x_displacement)*scaleFactor), int((k[1]+y_displacement)*scaleFactor))	
				new_keypoint = ((k[0]+x_displacement)*scaleFactor, (k[1]+y_displacement)*scaleFactor)	
		
			keypoints[i] = new_keypoint
			#newKeypoints.append(new_keypoint)
	
	return keypoints
'''
def poseIsConfident(keypoints, thresholdNoneBelow, thresholdNotMoreThanNBelow, N):
	
	nBelowThreshold = 0
	for i, k in enumerate(keypoints):
		if k[2] < thresholdNoneBelow:
			return False
		if k[2] < thresholdNotMoreThanNBelow:
			nBelowThreshold = nBelowThreshold+1
			if nBelowThreshold>N:
				return False
	return True

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

'''def copyKeypoints(keypoints):
	newKeypoints = [] 
	for i, k in enumerate(keypoints):
		new_keypoint = (k[0], k[1], k[2])	
		newKeypoints.append(new_keypoint)
	return newKeypoints
'''
