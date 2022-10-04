import cv2
import numpy as np
import poseUtils
import openPoseUtils
import normalization_heatmaps
import util_viz
import math

normalizer = normalization_heatmaps.NormalizationHeatmaps(outputRes=128, sigma=2)
        
#ORIGINAL	
keypoints = openPoseUtils.json2Keypoints('dynamicData/012_keypoints.json', True)
#referenceBoneIndex, dummy = openPoseUtils.reference_bone(keypoints)
#visualize(keypoints, 1000, 1000)

#NORMALIZED (HEATMAPS)
poseHeatmaps, scaleFactor, x_displacement, y_displacement = normalizer.normalize(keypoints)
print("poseHeatmaps[0]: ",poseHeatmaps[0][0])
print("poseHeatmaps shape: ",poseHeatmaps.shape)
#poseHeatmapsDisplay = poseHeatmaps.transpose(-1, 0, 1)
print("poseHeatmaps shape: ",poseHeatmaps.shape)
cv2.imshow("hm"+str(0), poseHeatmaps[0])

#DENORMALIZED
print(poseHeatmaps.shape)
print("poseHeatmaps.shape: ", poseHeatmaps.shape)
print(poseHeatmaps)
keypoints = normalization_heatmaps.denormalize(poseHeatmaps, scaleFactor, x_displacement, y_displacement)
normalization_heatmaps.visualize(keypoints, 1000, 1000)



