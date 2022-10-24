'''
- 3D to 2D perspective projection
- https://towardsdatascience.com/camera-extrinsic-matrix-with-example-in-python-cfe80acab8dd
'''	

#IMPORTANT: We asume childFromPoint = parentToPoint

from numpy.linalg import inv
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import pyrender
import math
import perspective_projection_extrinsics as pp
import BodyModelOPENPOSE15
import openPoseUtils

#BodyModelOPENPOSE15.POSE_BODY_25_BODY_PARTS_DICT  
#BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP contains the pairs
#[parent_joint, children_joint]
#14 bones
#POSE_BODY_25_PAIRS_RENDER_GP[13] is the neck and will be our reference
#13 = 1,0 neckJ->noseJ

#calcular fills de (joint=1) bone=[1,0] (HARCODED)
#busquem bones que tinguin 1 com a primer:
        #e.g. [1,8] (següent a cercar serà el 8)
        #calulem l'angle pare-fill
        #calcular fills de (joint=8) bone=[1,8] 
        #busquem bones que tinguin 8 com a primer:
            #e.g. [8,9] (següent a cercar serà el 9)
            #calulem l'angle pare-fill
            #calcular fills de (9) el pare és [8,9]

'''
#These parts define a tree as they are [parent, child] being the neck (1) the main parent
#POSE_BODY_25_PAIRS_RENDER_GP = [[1,8],   [1,2],   [1,5],   [2,3],   [3,4],   [5,6],   [6,7],   [8,9],   [9,10],  [10,11], [8,12],  [12,13], [13,14],  [1,0]]

def processChildrenBones(joint, parentBone, resultsList):
    #The result should be this
    resultsList = angles relative to their parent of ([1,8], [8,9], [9,10], [10,11], [8,12], [12,13], [13, 14], [1,2], [2,3], [3,4], [1,5], [5,6], [7,7]
    #Does not include information about neck bone
    #to reconstruct need to keep the position of joint 1

    for b in BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP:
        #arrenca de joint i no és ell mateix
        if b[0] == joint and (b[0] != parentBone[0] and b[1] != parentBone[1]):
            compact_angle = relativeAngle(childFromPoint, childToPoint, parentFromPoint, parentToPoint)
            save angle
            processChildrenBones(b[1], b, resultsList)
    
   

def reconstructPose(anglelist):
    bone 1 = reconstruct bone 1
    reconstructChildrens(bone 1)

def reconstructChildrens(parent [1,0], anglelist)
    buscar en anglelist bones con parent bone [1,0]
        bonechild = childVectorFromParentVectorAndRelativeAngle(parentFromPoint, parentToPoint, compact_axis_angle, childVectorMagnitude):



processChildrenBones(1, [1,0], resultsList)
'''

#These parts define a tree as they are [parent, child] being the neck (1) the main parent
#POSE_BODY_25_PAIRS_RENDER_GP = [[1,8],   [1,2],   [1,5],   [2,3],   [3,4],   [5,6],   [6,7],   [8,9],   [9,10],  [10,11], [8,12],  [12,13], [13,14],  [1,0]]
def processChildrenBones(jointNumber, parentBonePair, keypoints):
    #The result should be this
    #resultsList = angles relative to their parent of ([1,8], [8,9], [9,10], [10,11], [8,12], [12,13], [13, 14], [1,2], [2,3], [3,4], [1,5], [5,6], [7,7]
    #Does not include information about neck bone
    #to reconstruct need to keep the position of joint 1
    for b in BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP:
        #arrenca de joint i no és ell mateix
        #print("checking ",b)
        if b[0] == jointNumber and (b[0] != parentBonePair[0] or b[1] != parentBonePair[1]):
            childFromPoint=keypoints[b[0]]
            childToPoint=keypoints[b[1]]
            parentFromPoint=keypoints[parentBonePair[0]]
            parentToPoint=childFromPoint
            #compact_axis_angle = relativeAngle(childFromPoint, childToPoint, parentFromPoint, parentToPoint)
            print("parentBonePair: ", parentBonePair)
            print("childBonePair: ", b)
            #print("angle: ", compact_axis_angle)
            processChildrenBones(b[1], b, keypoints)
    

  
def angleTwoVectors(vector1, vector2):
    #θ = cos-1 [ (a · b) / (|a| |b|) ]
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    angle = np.arccos(np.dot(unit_vector1, unit_vector2)) #angle in radian
    return angle

def rotationBoneAWithRespectToBoneB(vector1, vector2):
    """
    Computes the rotation that would enable to align boneA to boneB.
    Works on the global axis (the coordinates are global).
    The resulting axis+angle rotation cannot be applied directly in pose mode
    because in pose mode works with local axis
    """
    rot_angle =  angleTwoVectors(vector1, vector2)
    rot_axis = np.cross(vector1, vector2)
    return rot_axis, rot_angle

def rotation_matrix(axis, angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotateVectorAxisangle(vector, rot_angle, rot_axis):
    #assumes unit vector I guess
    mrot_aux = rotation_matrix(rot_axis, rot_angle)
    rotatedVector = np.dot(mrot_aux, vector)
    return rotatedVector

def axisangleCompact(rot_angle, rot_axis):
    #divide vector by its norm (lenght)
    rot_axis_unit = rot_axis / np.linalg.norm(rot_axis)

    #and multiply for the angle
    rot_axis_with_angle_as_magnitude = rot_axis_unit * rot_angle
    return rot_axis_with_angle_as_magnitude

def axisangleCompact2axisAngle(rot_axis_with_angle_as_magnitude):
    #divide vector by its norm (lenght)
    rot_angle = np.linalg.norm(rot_axis_with_angle_as_magnitude)

    rot_axis_unit = rot_axis_with_angle_as_magnitude / rot_angle
    
    return rot_axis_unit, rot_angle

def relativeAngle(childFromPoint, childToPoint, parentFromPoint, parentToPoint):
	childVector = childToPoint-childFromPoint
	parentVector = parentToPoint-parentFromPoint
	rot_axis, rot_angle = rotationBoneAWithRespectToBoneB(parentVector, childVector)
	compact_axis_angle = axisangleCompact(rot_angle, rot_axis)
	return compact_axis_angle

def childVectorFromParentVectorAndRelativeAngle(parentFromPoint, parentToPoint, compact_axis_angle, childVectorMagnitude):
	rot_axis, rot_angle = axisangleCompact2axisAngle(compact_axis_angle)
	#vector from the origin (0,0)
	parentVector = parentToPoint-parentFromPoint
	parentVectorMagnitude = np.linalg.norm(parentVector)
	childVector = parentVector/parentVectorMagnitude
	childVector = childVector * childVectorMagnitude
	childVector = rotateVectorAxisangle(childVector, rot_angle, rot_axis)
	childToPoint = parentFromPoint
	childFromPoint = childToPoint-childVector
	return childFromPoint, childToPoint

#Matplotlib 10x10 image
pixel_plot, ax = plt.subplots()
plt.rcParams["figure.figsize"] = [10, 10]
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.grid()

path =  "dynamicData/H36Mtest_original_v2_noreps/100.json"
keypoints = openPoseUtils.json2Keypoints(path) 
keypoints, dummy = openPoseUtils.removeConfidence(keypoints)
keypoints = np.array(keypoints) 
print(keypoints)
processChildrenBones(1, [1,0], keypoints)

#A cube
points3d = np.array([[1,1,1], [1,5,1], [5,5,1], [5,1,1],  #fron square
	                 [1,1,5], [1,5,5], [5,5,5], [5,1,5]]) #back square

#Lines between the 3d points
lines = np.array([[0,1],[1,2],[2,3],[3,0], #fron square
	             [4,5],[5,6],[6,7],[7,4], #back square
	             [0,4],[1,5],[2,6],[3,7]]
	             )

#3D pose -> lenghts
for l, line in enumerate(lines):
    from_3dpoint = points3d[line[0]]
    to_3dpoint = points3d[line[1]]
    squared_dist = np.sum((from_3dpoint-to_3dpoint)**2, axis=0)
    dist = np.sqrt(squared_dist)
    print("dist=", dist)

#3D pose -> angles

#we first compute the two vectors
#they share a point but this does not matter to compute their angle (will be useful to reconstruct)

#line0
childFromPoint = points3d[lines[0][0]]
childToPoint = points3d[lines[0][1]]

print("original childFromPoint=", childFromPoint)
print("original childToPoint=", childToPoint)

#line1
parentFromPoint = points3d[lines[1][0]]
parentToPoint = points3d[lines[1][1]]

#previous information
childVector = childToPoint-childFromPoint
childVectorMagnitude = np.linalg.norm(childVector)

#obtain representation
compact_axis_angle = relativeAngle(childFromPoint, childToPoint, parentFromPoint, parentToPoint)

#back to 3D point
childFromPoint, childToPoint = childVectorFromParentVectorAndRelativeAngle(parentFromPoint, parentToPoint, compact_axis_angle, childVectorMagnitude)
print("reconstructed childFromPoint=", childFromPoint)
print("reconstructed childToPoint=", childToPoint)

######################################
#Apply camera pose (extrinsics)
rotation_angles = [np.pi/4]
rotation_order = 'z'#'y'
translation_offset = np.array([7, 2, 0])
points3d_cam = pp.points3d_relative_to_camera(points3d, rotation_angles, rotation_order, translation_offset)

#Apply camera intrinsics
focal_pane_width=10
focal_pane_height=10
focal_distance = 0.8#1
scalex = 1
scaley = 1
ppx=focal_pane_width/2
ppy=focal_pane_height/2
points2d = pp.points3d_cam_to_pixels(points3d_cam, focal_distance, ppx, ppy, scalex, scaley)

#Show points 2d
for i, pixel in enumerate(points2d):
	plt.plot(pixel[0],pixel[1], marker='o', markersize=5, markeredgecolor="red", markerfacecolor="red") 
	ax.annotate(str(i), (pixel[0], pixel[1]))
	print("point3d("+str(i)+"):"+str(pixel[0])+","+str(pixel[1]))

#Show lines
for l, line in enumerate(lines):
	from_pixel = points2d[line[0]]
	to_pixel = points2d[line[1]]
	plt.plot([from_pixel[0],to_pixel[0]], [from_pixel[1],to_pixel[1]], linewidth=1, markersize=0)

#plt.show()
plt.savefig('perspective_projection_extrinsics.png')

