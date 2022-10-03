
import json
import cv2
import numpy as np
import math
import sys

def debugKeypoints(keypoints):
	print("----- debug keypoints -----")
	print("Type: ", type(keypoints))
	print("Len: ", len(keypoints))
	print("Sample: ", keypoints[0])
	print("---------------------------")

def keypoints2Numpy(keypoints):
	keypointsNP = np.vstack(keypoints)
	return keypointsNP

def keypointsListFlatten(keypoints):
	keypointsNP = np.vstack(keypoints)
	keypointsNPflat = keypointsNP.flatten()
	keypointsNPflatFloat = [float(k) for k in keypointsNPflat]
	return keypointsNPflatFloat

def deflatten(keypointsFlat, haveConfidence):
	if haveConfidence: 
		keypoints = list(zip(
			list(map(float, keypointsFlat[0::3])), 
			list(map(float, keypointsFlat[1::3])), 
			list(map(float, keypointsFlat[2::3]))  
		))
	else:
		keypoints = list(zip(
			list(map(float, keypointsFlat[0::2])), 
			list(map(float, keypointsFlat[1::2]))	        
		))
	return keypoints

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
    img = cv2.line(img, (int(keypoint1[0]), int(keypoint1[1])), (int(keypoint2[0]), int(keypoint2[1])), (color[2], color[1], color[0]), thickness, lineType=cv2.LINE_AA)
    
def draw_pose(img, keypoints, threshold, keypoint_index_pairs, colors, haveThreshold, thickness=2):        
	""" 
	Draws keypoints (if haveThreshold with a confidence > threshold) over img
	"""
	for boneNumer, bone in enumerate(keypoint_index_pairs):
		#print("Part: "+str(bone[0])+","+str(bone[1]))
		#print("bone: ", bone)
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

def draw_pose_scaled_centered(img, keypoints, threshold, keypoint_index_pairs, colors, haveThreshold, scaleFactor, centerX, centerY, centerKeypointIndex, thickness=1):        	
	height, width = img.shape[:2]
	#print("width=", width)
	#scaleFactor = 23
	keypoints_scaled = scale(keypoints, scaleFactor) #4 if image 64x64, 23 if 128x128
	keypoints_scaled_centered, dummy, dummy = center_pose(keypoints_scaled, centerX, centerY, centerKeypointIndex)
	draw_pose(img, keypoints_scaled_centered, threshold, keypoint_index_pairs, colors, haveThreshold, thickness)        

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

'''
def magnitude(keypoints, idx1, idx2)
	keypoint1 = keypoints[idx1]
	keypoint2 = keypoints[idx2]
	x_distance = keypoint1[0]-keypoint2[0]
	y_distance = keypoint1[1]-keypoint2[1]
	magnitude = math.sqrt(pow(x_distance, 2)+pow(y_distance, 2))
	return magnitude
'''

def center_pose(keypoints, centerX, centerY, centerKeypointIndex):
	#Center the reference keypoint at centerX and centerY 

	keypoint1 = keypoints[centerKeypointIndex]
	centered_keypoints = keypoints.copy()
	x_displacement = keypoint1[0] - centerX
	y_displacement = keypoint1[1] - centerY
	for i, k in enumerate(centered_keypoints):
		if k[0] != 0 or k[1] != 0:
			k[0] = k[0]-x_displacement
			k[1] = k[1]-y_displacement
	return centered_keypoints, x_displacement, y_displacement

def scale_pose(keypoints, keypoint_index_pairs, boneSpineIndex, spinesize):
	scaled_keypoints = keypoints.copy()

	keypoint1 = keypoints[keypoint_index_pairs[boneSpineIndex][0]]
	keypoint2 = keypoints[keypoint_index_pairs[boneSpineIndex][1]]

	x_distance = keypoint1[0]-keypoint2[0]
	y_distance = keypoint1[1]-keypoint2[1]

	#Compute the length of the spine
	magnitudeSpine = math.sqrt(pow(x_distance, 2)+pow(y_distance, 2))

	#Compute scale factor to obtain the desired spine size
	scaleFactor = magnitudeSpine/spinesize

	scaled_keypoints = scale(scaled_keypoints, scaleFactor)
	
	return scaled_keypoints, scaleFactor

def scale(keypoints, scaleFactor):
	scaled_keypoints = keypoints.copy()

	#Scale: dividing any keypoint by the scale factor
	for i, k in enumerate(scaled_keypoints):
		if k[0] != 0 or k[1] != 0:
			k[0] = k[0]/scaleFactor
			k[1] = k[1]/scaleFactor

	return scaled_keypoints

def normalize_poseTODO(keypoints, keypoint_index_pairs, spinesize, width, height, boneSpineIndex, keepThreshold):
		
	#MALAMENT: CAl primer escalar i després centrar! 

	keypointsNP = keypoints2Numpy(keypoints)
	#center (the nose)
	#center (to 0,0 for training/inference or to width,height for drawing)
	centered_keypoints, x_displacement, y_displacement = center_pose(keypointsNP, 0, 0, 0)
	
	#scaling (relative to the given reference bone size)
	normalized_keypoints, scaleFactor = scale_pose(centered_keypoints, keypoint_index_pairs, boneSpineIndex, spinesize)

	return normalized_keypoints, scaleFactor, x_displacement, y_displacement

def denormalize_poseTODO(keypoints, scaleFactor, x_displacement, y_displacement, keepThreshold):
	keypointsNP = keypoints2Numpy(keypoints)

	denormalized_keypoints = scale(keypointsNP, 1/scaleFactor)
	
	for i, k in enumerate(denormalized_keypoints):
		k[0] =  k[0]+x_displacement
		k[1] =  k[1]+y_displacement

	return denormalized_keypoints


def normalize_pose(keypoints, keypoint_index_pairs, spinesize, width, height, boneSpineIndex, keepThreshold):
	
	#WARNING: When zero the values should remain zero, also in denormalize
	#The values will be replaced later but they may impact the process

	normalized_keypoints = keypoints.copy()

	#print("normalizing, normalized_keypoints[10]=", normalized_keypoints[10])

	#computing spine distance
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
		#scaleFactor = magnitudeSpine/1

		#Scale: dividing any keypoint by the scale factor
		for i, k in enumerate(normalized_keypoints):
			if keepThreshold:
				#new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor), k[2]) 
				new_keypoint = (k[0]/scaleFactor, k[1]/scaleFactor, k[2]) 
			else: 
				#new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor)) 
				new_keypoint = (k[0]/scaleFactor, k[1]/scaleFactor) 
			normalized_keypoints[i] = new_keypoint

		#Center to the top of the nose
		keypoint1 = normalized_keypoints[keypoint_index_pairs[boneSpineIndex][1]]
		#centerX = width/2
		#centerY = height/2
		centerX = 0.5
		centerY = 0.5
		x_displacement = keypoint1[0] - centerX
		y_displacement = keypoint1[1] - centerY
		for i, k in enumerate(normalized_keypoints):

			if keepThreshold:
				#WARNING: When both values are zero it means 
				#that we did a removeConfidence before
				#DECISSION: Keep zeros instead of center them
				if (k[0] != 0 and k[0] != 0):
					new_keypoint = (k[0]-x_displacement, k[1]-y_displacement, k[2]) 
				else:
					new_keypoint = (k[0], k[1], k[2]) 
			else:
				#if k[0]!=0 or k[1]!=0:
				if (k[0] != 0 and k[0] != 0):
					new_keypoint = (k[0]-x_displacement, k[1]-y_displacement)
				else:
					new_keypoint = (k[0], k[1])
			normalized_keypoints[i] = new_keypoint

	else:
		raise Exception('Reference bone keypoints are zero :-(')


	return normalized_keypoints, scaleFactor, x_displacement, y_displacement

'''
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
		centerX = 0.5
		centerY = 0.5
		x_displacement = keypoint1[0] - centerX
		y_displacement = keypoint1[1] - centerY
		#Scale: dividing any keypoint by the scale factor
		for i, k in enumerate(normalized_keypoints):
			if keepThreshold:
				#new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor), k[2]) 
				new_keypoint = ((k[0]-keypoint1[0])/scaleFactor + centerX, (k[1]-keypoint1[1])/scaleFactor + centerY, k[2]) 
			else: 
				#new_keypoint = (int(k[0]/scaleFactor), int(k[1]/scaleFactor)) 
				new_keypoint = ((k[0]-keypoint1[0])/scaleFactor + centerX, (k[1]-keypoint1[1])/scaleFactor + centerY) 
			normalized_keypoints[i] = new_keypoint
	else:
		raise Exception('Reference bone keypoints are zero :-(')


	return normalized_keypoints, scaleFactor, x_displacement, y_displacement
'''


def denormalize_pose(keypoints, scaleFactor, x_displacement, y_displacement, keepThreshold):
	#WARNING: It skips 0 values in the normalized pose
	#but this means that you cannot normalize around center 0
	newKeypoints = [] 
	for i, k in enumerate(keypoints):
		if scaleFactor != -1:
			if keepThreshold:
				#new_keypoint = (int((k[0]+x_displacement)*scaleFactor), int((k[1]+y_displacement)*scaleFactor), k[2]) 
				if k[0]!=0 and k[1]!=0: #to avoid denormalizing zeros?
					new_keypoint = ((k[0]+x_displacement)*scaleFactor, (k[1]+y_displacement)*scaleFactor, k[2]) 
				else:
					new_keypoint = (0 ,0, 0.0)
			else: 
				if k[0]!=0 and k[1]!=0:
				#new_keypoint = (int((k[0]+x_displacement)*scaleFactor), int((k[1]+y_displacement)*scaleFactor))	
					new_keypoint = ((k[0]+x_displacement)*scaleFactor, (k[1]+y_displacement)*scaleFactor)	
				else:
					new_keypoint = (0 ,0)
			#keypoints[i] = new_keypoint
			newKeypoints.append(new_keypoint)
	return newKeypoints

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
