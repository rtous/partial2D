import cv2
from SimpleHigherHRNet import SimpleHigherHRNet
from misc.visualization import draw_skeleton, draw_points_and_skeleton, joints_dict
import json
import numpy as np
import pathlib
from os import listdir
from os.path import isfile, join, splitext

HRNET_COCO_DICT = [
                "Nose",
                "LEye",
                "REye",
                "LEar",
                "REar",
                "LShoulder",
                "RShoulder",
                "LElbow",
                "RElbow",
                "LWrist",
                "RWrist",
                "LHip",
                "RHip",
                "LKnee",
                "RKnee",
                "LAnkle",
                "RAnkle"
            ]

POSE_BODY_25_BODY_PARTS = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel" #, "Background"
]

def dummyKeypoints(howMany):
    dummy = []
    for i in range(howMany):
        new_keypoint = (0, 0, 0.0)
        dummy.append(new_keypoint)
    dummy = np.array(dummy).flatten()
    return dummy

def midkeypoint(k1, k2):
    return ((k1[0]+k2[0])/2, (k1[1]+k2[1])/2, (k1[2]+k2[2])/2)

def flipkeypoint(k): 
    #hrnet uses y,x coordinates?
    return (k[1], k[0], k[2])

def hrnetKeypointsInOpenPoseOrder(keypoints):
    newKeypoints = []  
    #Copy common keypoints
    for keypoint_name in POSE_BODY_25_BODY_PARTS:
        if keypoint_name in HRNET_COCO_DICT:
            hrnet_index = HRNET_COCO_DICT.index(keypoint_name)
            new_keypoint = (keypoints[hrnet_index][0], keypoints[hrnet_index][1], keypoints[hrnet_index][2])
        elif keypoint_name == "MidHip":
            hrnet_index_LHip = HRNET_COCO_DICT.index("LHip")
            hrnet_index_RHip = HRNET_COCO_DICT.index("RHip")
            new_keypoint = midkeypoint(keypoints[hrnet_index_LHip], keypoints[hrnet_index_RHip])
        elif keypoint_name == "Neck":
            hrnet_index_LShoulder = HRNET_COCO_DICT.index("LShoulder")
            hrnet_index_RShoulder = HRNET_COCO_DICT.index("RShoulder")
            new_keypoint = midkeypoint(keypoints[hrnet_index_LShoulder], keypoints[hrnet_index_RShoulder])
        else:
            new_keypoint = (0.0, 0.0, 0.0)

        new_keypoint = flipkeypoint(new_keypoint)

        newKeypoints.append(new_keypoint)

    return newKeypoints

def hrnet2openpose(hrntKeypoints, path):

	keypoints = hrnetKeypointsInOpenPoseOrder(hrntKeypoints)

	keypointsFlat = np.array(keypoints).flatten()

	data = {"version": 1.3, "people": [{"person_id": [-1], "pose_keypoints_2d": [], "pose_keypoints_3d": [], "face_keypoints_3d": [], "face_keypoints_2d": [], "hand_left_keypoints_3d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_3d": [], "hand_right_keypoints_2d": []}]}
	
	data['people'][0]['pose_keypoints_2d'] = keypointsFlat.tolist()
	data['people'][0]['face_keypoints_2d'] = dummyKeypoints(70).tolist()
	data['people'][0]['hand_right_keypoints_2d'] = dummyKeypoints(21).tolist()
	data['people'][0]['hand_left_keypoints_2d'] = dummyKeypoints(21).tolist()

	outfile = open(path, 'w')
	json.dump(data, outfile)
	outfile.close()


model = SimpleHigherHRNet(32, 17, "./weights/pose_higher_hrnet_w32_512.pth")

INPUTPATHIMAGES = "input"
OUTPUTPATH = "output"

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

imageFiles = [f for f in listdir(INPUTPATHIMAGES) if isfile(join(INPUTPATHIMAGES, f)) and (f.endswith(".png") or f.endswith(".jpg"))]
for filename_image in imageFiles:

    filename_noextension = splitext(filename_image)[0]
    filename_keypoints = filename_noextension+"_keypoints.json"

   
    originalImagePath = join(INPUTPATHIMAGES, filename_image)

    image = cv2.imread(originalImagePath, cv2.IMREAD_COLOR)

    joints = model.predict(image)

    print("{:.2f}".format(joints[0][10][2]))

    image = draw_points_and_skeleton(image, joints[0], joints_dict()['coco']['skeleton'])

    targetImagePath = join(OUTPUTPATH, filename_noextension+"_pose.jpg")

    cv2.imwrite(targetImagePath, image)

    targetJSONPath = join(OUTPUTPATH, filename_keypoints)

    hrnet2openpose(joints[0], targetJSONPath)

