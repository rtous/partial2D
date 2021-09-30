
import json
import cv2
import numpy as np


def draw_pose(img, keypoints, threshold=0.2):

    for idx, k in enumerate(zip(keypoints[0::3], keypoints[1::3], keypoints[2::3])):
        x = k[0]
        y = k[1]
        c = k[2]
        img = cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)

        '''


    xys = {}
    for label, keypoint in keypoints:
        #if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        img = cv2.circle(img, (int(keypoint.yx[1]), int(keypoint.yx[0])), 5, (0, 255, 0), -1)

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        img = cv2.line(img, (ax, ay), (bx, by), (0, 255, 255), 2) 
        '''
 
# Opening JSON file
f = open('001_keypoints.json',)
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
person = data['people'][0]

keypoints = person['pose_keypoints_2d']
for idx, k in enumerate(zip(keypoints[0::3], keypoints[1::3], keypoints[2::3])):
    x = k[0]
    y = k[1]
    c = k[2]
    print (str(x) +',' + str(y) + '(' + str(c) + ')')

blank_image = np.zeros((500,500,3), np.uint8)

draw_pose(blank_image, keypoints, threshold=0.2)

#cv2.drawKeypoints(blank_image, kps5, img_brisk, kps5, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
cv2.imshow('Test', blank_image)
cv2.resizeWindow('Test', 600,600)

k = cv2.waitKey(0) & 0xFF

if k == 27:         # wait for ESC key to exit
  cv2.destroyAllWindows()
else:
  print("Exit")


 
# Closing file
f.close()