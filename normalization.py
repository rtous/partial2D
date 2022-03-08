import cv2
import numpy as np
import poseUtils
import openPoseUtils
import util_viz

#https://elte.me/2021-03-10-keypoint-regression-fastai

#from https://github.com/rohitrango/Adversarial-Pose-Estimation/blob/master/datasets/mpii.py

HEATMAP_WIDTH = 64#128#64 
HEATMAP_HEIGHT = 64#128#64

def GetTransform(center, scale, rot, res):
	h = scale
	t = np.eye(3)

	t[0, 0] = res / h
	t[1, 1] = res / h
	t[0, 2] = res * (- center[0] / h + 0.5)
	t[1, 2] = res * (- center[1] / h + 0.5)

	if rot != 0:
		rot = -rot
		r = np.eye(3)
		ang = rot * np.math.pi / 180
		s = np.math.sin(ang)
		c = np.math.cos(ang)
		r[0, 0] = c
		r[0, 1] = - s
		r[1, 0] = s
		r[1, 1] = c
		t_ = np.eye(3)
		t_[0, 2] = - res / 2
		t_[1, 2] = - res / 2
		t_inv = np.eye(3)
		t_inv[0, 2] = res / 2
		t_inv[1, 2] = res / 2
		t = np.dot(np.dot(np.dot(t_inv,  r), t_), t)

	return t


def transform(pt, center, scale, rot, res, invert = False):
	pt_ = np.ones(3)
	pt_[0], pt_[1] = pt[0], pt[1]

	t = GetTransform(center, scale, rot, res)
	if invert:
		t = np.linalg.inv(t)
	new_point = np.dot(t, pt_)[:2]
	new_point = new_point.astype(np.int32)
	return new_point

def drawGaussian(img, pt, sigma, truesigma=-1):
	y, x = img.shape[:2]
	xx, yy = np.meshgrid(np.arange(y), np.arange(x))
	xx = xx*1.0
	yy = yy*1.0
	img = np.exp(-((xx - pt[0])**2 + (yy - pt[1])**2)/(2*sigma*sigma))
	return img

def keypointsToHeatmaps(keypoints, outputRes, sigma = 2):
	nJoints = len(keypoints)
	#inp = I.Crop(img, c, s, r, self.inputRes) / 255.
	#We specify dtype="float32" or pytorch will complain
	out = np.zeros((nJoints, outputRes, outputRes), dtype="float32")
	for i in range(nJoints):
		#keypoints[i] = transform(keypoints[i], [0.1, 0.1], 4000, 0, outputRes)
		if keypoints[i][0] != 0 or keypoints[i][1] != 0:
			out[i] = drawGaussian(out[i], keypoints[i], sigma, 0.5 if outputRes==32 else -1)
	
	#out = out.transpose(1, 2, 0)
	#print("keypointsToHeatmaps out", out.shape)
	return out

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def heatmapsToKeypoints(heatmaps):
#https://stackoverflow.com/questions/60032705/how-to-parse-the-heatmap-output-for-the-pose-estimation-tflite-model
	
	heatmaps = heatmaps.transpose(1, 2, 0)
	
	#heatmaps = np.rollaxis(heatmaps, -1, 0)
	#print("heatmaps shape:", heatmaps.shape)
	scores = sigmoid(heatmaps)
	num_keypoints = scores.shape[2]
	#print("num_keypoints:", num_keypoints)
	#heatmap_positions = []
	heatmap_positions = np.zeros((num_keypoints, 2))
	offset_vectors = []
	confidences = []
	for ki in range(0, num_keypoints ):
	    x,y = np.unravel_index(np.argmax(scores[:,:,ki]), scores[:,:,ki].shape)
	    #print("x,y=%d,%d" % (x,y))
	    #confidences.append(scores[x,y,ki])
	    #offset_vector = (offsets[y,x,ki], offsets[y,x,num_keypoints+ki])
	    #heatmap_positions.append((y,x))
	    heatmap_positions[ki][0] = y
	    heatmap_positions[ki][1] = x
	    #offset_vectors.append(offset_vector)
	#image_positions = np.add(np.array(heatmap_positions) * output_stride, offset_vectors)
	
	#for ip in image_positions:

	#	keypoints = [KeyPoint(i, pos, confidences[i]) for i, pos in enumerate(image_positions)]
	
	return heatmap_positions


#def image(keypoints, keepConfidence=True):

def normalize(keypoints, keepConfidence=True):
	#Input: keypoints as list
	#Output: heatmaps as numpy arrays (25, outputRes, ouputRes)
	scaleFactor = 1000/HEATMAP_WIDTH
	keypointsNP = poseUtils.keypoints2Numpy(keypoints)
	keypoints = poseUtils.scale(keypointsNP, scaleFactor)
	keypoints, x_displacement, y_displacement = poseUtils.center_pose(keypoints, HEATMAP_WIDTH/2, HEATMAP_WIDTH/4, 0)
	#Debug
	#visualize(keypoints, outputRes, outputRes)
	poseHeatmaps = keypointsToHeatmaps(keypoints, HEATMAP_WIDTH)
	#print(poseHeatmaps.shape)
	return poseHeatmaps, scaleFactor, x_displacement, y_displacement

def denormalize(heatmaps, scaleFactor, x_displacement, y_displacement):
	#Input: heatmaps as numpy arrays (25, outputRes, ouputRes)
	#Ouput: keypoints as np array
	keypoints = heatmapsToKeypoints(heatmaps)
	for i, k in enumerate(keypoints):
		k[0] = (k[0]+x_displacement)*scaleFactor
		k[1] = (k[1]+y_displacement)*scaleFactor 
	return keypoints

def denormalizeBatch(batch, scaleFactor, x_displacement, y_displacement):
	denormalized = []
	for i in range(batch.shape[0]):
		#print("batch[i].shape: ", batch[i].shape)
		denormalized.append(denormalize(batch[i], scaleFactor[i], x_displacement[i], y_displacement[i]))
	return denormalized
			
def visualize(keypoints, width, height):
	blank_image = np.zeros((width,height,3), np.uint8)
	poseUtils.draw_pose(blank_image, keypoints, 0.1, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
	poseUtils.displayImage(blank_image, width, height)

#Useful for debugging
def nullPoseBatch(nJoints, outputRes, batchsize):
	nullPoseBatch = np.zeros((batchsize, nJoints, outputRes, outputRes), dtype="float32")
	return nullPoseBatch

#Useful for debugging
def onesConfidentValuesBatch(nJoints, batchsize):
	onesConfidentValuesBatch = np.ones((batchsize, nJoints), dtype="float32")
	return onesConfidentValuesBatch



'''
#ORIGINAL	
keypoints = openPoseUtils.json2Keypoints('dynamicData/012_keypoints.json')
referenceBoneIndex, dummy = openPoseUtils.reference_bone(keypoints)
#visualize(keypoints, 1000, 1000)

#NORMALIZED (HEATMAPS)
poseHeatmaps, scaleFactor, x_displacement, y_displacement = normalize(keypoints, HEATMAP_WIDTH)
print("poseHeatmaps[0]: ",poseHeatmaps[0][0])
print("poseHeatmaps shape: ",poseHeatmaps.shape)
#poseHeatmapsDisplay = poseHeatmaps.transpose(-1, 0, 1)
print("poseHeatmaps shape: ",poseHeatmaps.shape)
cv2.imshow("hm"+str(0), poseHeatmaps[0])

#DENORMALIZED
print(poseHeatmaps.shape)
keypoints = denormalize(poseHeatmaps, scaleFactor, x_displacement, y_displacement)
visualize(keypoints, 1000, 1000)
'''


