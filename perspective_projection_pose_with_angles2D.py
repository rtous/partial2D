'''
- Normalize a 2D pose as relative angles among parts (bones)
- In 2D we need the lengths (as they change) but this way we are rotation independent
- The normalized pose includes the angles, the lenghts and the position of the neck joint
- The neck->nose bone is the reference bone and the neck joint the starting point.
- It's angle will be the angle wrt the X axis 
- All bones starting from the neck are CHILDREN and we save the angle with respect to the parent (the neck), and their length
- The POSE_BODY_25_PAIRS_RENDER_GP paris are exactly what we need
- They are pairs [fromparent, tochildren]
- The angles and lenths are stored in vector of same shape as keypoints
- In joint K we store the angle of children [?, K] to its parent bone.

- VALUES NORMALIZATION:
    - lenths are normalized with respect to the neck (length 1)
    - angles are normalized as ratio to 2pi


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
#import perspective_projection_extrinsics as pp
import BodyModelOPENPOSE15
import openPoseUtils

########### INFO ###################

#BodyModelOPENPOSE15.POSE_BODY_25_BODY_PARTS_DICT  
#BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP contains the pairs
#[parent_joint, children_joint]
#14 bones
#POSE_BODY_25_PAIRS_RENDER_GP[13] is the neck and will be our reference
#13 = 1,0 neckJ->noseJ

#These parts define a tree as they are [parent, child] being the neck (1) the main parent
#POSE_BODY_25_PAIRS_RENDER_GP = [[1,8],   [1,2],   [1,5],   [2,3],   [3,4],   [5,6],   [6,7],   [8,9],   [9,10],  [10,11], [8,12],  [12,13], [13,14],  [1,0]]

##################################################

#GLOBALS

rootJoint = 1 
rootBone = [1,0]


############################################################

def removeRootJoint(valuesWithRoot):
    #rootJoint (pos 1) has no angle neither length associated
    valuesWithoutRoot = np.zeros(len(BodyModelOPENPOSE15.POSE_BODY_25_BODY_PARTS_DICT)-1, dtype="float32")
    valuesWithoutRoot[0] = valuesWithRoot[0]
    for i in range(2, len(valuesWithRoot)):
        valuesWithoutRoot[i-1] = valuesWithRoot[i]
    return valuesWithoutRoot

def addRootJoint(valuesWithoutRoot):
    #normalization removes rootJoint (pos 1) as it has no angle neither length associated
    valuesWithRoot = np.zeros(len(BodyModelOPENPOSE15.POSE_BODY_25_BODY_PARTS_DICT), dtype="float32")
    valuesWithRoot[0] = valuesWithoutRoot[0]
    valuesWithRoot[1] = 0
    for i in range(1, len(valuesWithoutRoot)):
        valuesWithRoot[i+1] = valuesWithoutRoot[i]
    return valuesWithRoot

################ Representation functions ###################
def normalize(keypoints):
    #one angle for each joint
    #joint 0 need to be initialized with respect to the axis
    #joint 1 will not be used as is not terminal of any part
    
    angleList = np.zeros(len(BodyModelOPENPOSE15.POSE_BODY_25_BODY_PARTS_DICT ), dtype="float32")
    lengthList = np.zeros(len(BodyModelOPENPOSE15.POSE_BODY_25_BODY_PARTS_DICT ), dtype="float32")
    #rotation of the rootBone with respect to the horizontal axis

    angleList[0]=normalizeAngle(-relativeAngle(keypoints[rootBone[0]], keypoints[rootBone[1]], np.array([0,0]), np.array([1,0])))
    rootBoneVector = keypoints[rootBone[1]]-keypoints[rootBone[0]]
    rootBoneVectorLength = np.linalg.norm(rootBoneVector)

    if rootBoneVectorLength==0:
        raise Exception("cannot normalize if rootBoneVectorLength==0")

    #lengthList[0]=rootBoneVectorLength # without normalizing length
    lengthList[0]=1 #root bone will have lenght 1
    processChildrenBones(rootJoint, rootBone, keypoints, angleList, lengthList, rootBoneVectorLength) 
    

    angleList = removeRootJoint(angleList)
    lengthList = removeRootJoint(lengthList)

    angleListAndlengthList = np.concatenate((angleList, lengthList), axis=0)

    return angleListAndlengthList, keypoints[rootBone[0]], rootBoneVectorLength

   
def processChildrenBones(jointNumber, parentBonePair, keypoints, angleList, lengthList, rootBoneVectorLength):
    #The result should be this
    #resultsList = angles relative to their parent of ([1,8], [8,9], [9,10], [10,11], [8,12], [12,13], [13, 14], [1,2], [2,3], [3,4], [1,5], [5,6], [7,7]
    #Does not include information about neck bone
    #to reconstruct need to keep the position of joint 1
    for b in BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP:
        #arrenca de joint i no és ell mateix
        if b[0] == jointNumber and (b[0] != parentBonePair[0] or b[1] != parentBonePair[1]):
            childFromPoint=keypoints[b[0]]
            childToPoint=keypoints[b[1]]
            parentFromPoint=keypoints[parentBonePair[0]]
            parentToPoint=keypoints[parentBonePair[1]]
            '''
            print("-----------------")
            print("parentBonePair: ", parentBonePair)
            print("parentFromPoint: ", parentFromPoint)
            print("parentToPoint: ", parentToPoint)
            print("childBonePair: ", b)
            print("childFromPoint: ", childFromPoint)
            print("childToPoint: ", childToPoint)
            '''
            #saves the inverse angle to be able to directly apply it later
            angle = -relativeAngle(childFromPoint, childToPoint, parentFromPoint, parentToPoint)
            #print("angle: ", angle)
            #the angle is assigned to the end joint of the child
            angleList[b[1]]=normalizeAngle(angle)
            childVector = childToPoint-childFromPoint
            length = np.linalg.norm(childVector)

            if length==0:
                raise Exception("cannot normalize if length==0 (["+str(childToPoint)+"]-["+str(childFromPoint)+"])")

            #lengthList[b[1]]=length # without normalizing length
            lengthList[b[1]]=rootBoneVectorLength/length
            processChildrenBones(b[1], b, keypoints, angleList, lengthList, rootBoneVectorLength)

def angleTwoVectors(vector1, vector2):
    #can return negative angle 
    #angle that would align vector1 to vector2.
    #θ = cos-1 [ (a · b) / (|a| |b|) ]
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    #print("unit_vector1=",unit_vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    #print("unit_vector2=",unit_vector2)
    angle = math.atan2(np.dot(np.cross(vector1, vector2), 1), np.dot(vector1, vector2))
    #angle = atan2(norm(cross(a,b)), dot(a,b))
    #angle = math.atan2(np.linalg.norm(np.cross(vector1, vector2)), np.dot(vector1, vector2))
    return angle

def relativeAngle(childFromPoint, childToPoint, parentFromPoint, parentToPoint):
	childVector = childToPoint-childFromPoint
	parentVector = parentToPoint-parentFromPoint
	angle = angleTwoVectors(childVector, parentVector)
	return angle

def normalizeAngle(angle):
    #normalizes an angle between -pi and pi radians into positive 2pi and then divides by 2pi
    angle2pi = angle - 2 * np.pi * np.floor(angle / (2 * np.pi))
    return angle2pi/(2 * np.pi)
    return angle

def denormalizeAngle(normalizedAngle):
    angle2pi = normalizedAngle * 2 * np.pi
    if angle2pi > np.pi:
        angle = angle2pi - 2.0 * np.pi
    else:
        angle = angle2pi + 2.0 * np.pi
    return angle
    #return angle/2

#################### Reconstruction functions ####################
def denormalize(angleListAndlengthList, rootJointValue, rootBoneVectorLength):
    angleListAndlengthList = np.split(angleListAndlengthList, 2)
    angleList = angleListAndlengthList[0]
    lengthList = angleListAndlengthList[1]
    angleList = addRootJoint(angleList)
    lengthList = addRootJoint(lengthList)

    reconstructedKeypoints = np.zeros((len(BodyModelOPENPOSE15.POSE_BODY_25_BODY_PARTS_DICT), 2))

    originUnitVectorWithRootBoneAngle = rotateVector2D(np.array([1,0]), denormalizeAngle(angleList[rootBone[1]]))
    #originVectorWithRootBoneAngleAndMagnitude = originUnitVectorWithRootBoneAngle * lengthList[rootBone[1]] #without normalizing lenthgs

    originVectorWithRootBoneAngleAndMagnitude = originUnitVectorWithRootBoneAngle * (rootBoneVectorLength/lengthList[rootBone[1]]) 
    parentFromPoint = rootJointValue#keypoints[rootBone[0]] #need the rootJoint (is the offset)
    parentToPoint = originVectorWithRootBoneAngleAndMagnitude + parentFromPoint

    reconstructedKeypoints[rootBone[0]]=parentFromPoint
    reconstructedKeypoints[rootBone[1]]=parentToPoint

    #print("reconstructedKeypoints=", reconstructedKeypoints)

    reconstructChildrens(rootJoint, rootBone, angleList, lengthList, reconstructedKeypoints, rootBoneVectorLength)

    #print("reconstructedKeypoints=", reconstructedKeypoints)

    return reconstructedKeypoints

def rotateVector2D(vector, rot_angle):
    #assumes unit vector I guess
    mrot_aux = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
    rotatedVector = np.dot(mrot_aux, vector)
    return rotatedVector

def childVectorFromParentVectorAndRelativeAngle2D(parentFromPoint, parentToPoint, angle, childVectorMagnitude, childFromPointJointNumber, reconstructedKeypoints, rootBoneVectorLength):
    #vector from the origin (0,0)    
    parentVector = parentToPoint-parentFromPoint
    parentVectorMagnitude = np.linalg.norm(parentVector)
    
    originUnitVectorWithRootBoneAngle = rotateVector2D(parentVector/parentVectorMagnitude, angle)
    originVectorWithRootBoneAngleAndMagnitude = originUnitVectorWithRootBoneAngle * childVectorMagnitude

    childFromPoint = reconstructedKeypoints[childFromPointJointNumber]
    childToPoint = originVectorWithRootBoneAngleAndMagnitude + childFromPoint
    return childFromPoint, childToPoint

def reconstructChildrens(jointNumber, parentBonePair, angleList, lengthList, reconstructedKeypoints, rootBoneVectorLength):
    for b in BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP:
        #arrenca de joint i no és ell mateix
        if b[0] == jointNumber and (b[0] != parentBonePair[0] or b[1] != parentBonePair[1]):

            parentFromPoint=reconstructedKeypoints[parentBonePair[0]]
            parentToPoint=reconstructedKeypoints[parentBonePair[1]]

            angle = denormalizeAngle(angleList[b[1]])
            #childVectorMagnitude = lengthList[b[1]] # without normalizing length
            childVectorMagnitude = rootBoneVectorLength/lengthList[b[1]] 

            childFromPoint, childToPoint = childVectorFromParentVectorAndRelativeAngle2D(parentFromPoint, parentToPoint, angle, childVectorMagnitude, jointNumber, reconstructedKeypoints, rootBoneVectorLength)

            reconstructedKeypoints[b[0]]=childFromPoint
            reconstructedKeypoints[b[1]]=childToPoint

            reconstructChildrens(b[1], b, angleList, lengthList, reconstructedKeypoints, rootBoneVectorLength)


############################################################

#Test keypoints

'''
path =  "dynamicData/H36Mtest_original_v2_noreps/100.json"
keypoints = openPoseUtils.json2Keypoints(path) 
keypoints, dummy = openPoseUtils.removeConfidence(keypoints)
keypoints = np.array(keypoints) 
print(keypoints)

#normalize
angleListAndLengthList, rootJointValue, rootBoneVectorLength, dummy = normalize(keypoints)
print("angleListAndLengthList=", angleListAndLengthList)
print("angleListAndLengthList.shape=", angleListAndLengthList.shape)


#denormalize
reconstructedKeypoints = denormalize(angleListAndLengthList, rootJointValue, rootBoneVectorLength)

print(reconstructedKeypoints)
'''