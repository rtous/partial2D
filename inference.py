from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pytorchUtils
import argparse
from torchvision.datasets import MNIST
import pathlib
import json
from os import listdir
from os.path import isfile, join, splitext
import openPoseUtils
import poseUtils
import cv2
import traceback
import shutil
import sys
from torch.utils.tensorboard import SummaryWriter
import importlib
#import Configuration
#import models
import colors
import BodyModelOPENPOSE15
import BodyModelOPENPOSE25

mean=  466.20676
std=  114.26538

argv = sys.argv
try:
    DATASET_CROPPED=argv[1]
    DATASET_ORIGINAL=argv[2]
    OUTPUTPATH=argv[3]
    DATASET_CHARADE=argv[4]
    DATASET_CHARADE_IMAGES=argv[5]
    DATASET_TEST=argv[6]
    DATASET_TEST_IMAGES=argv[7]
    MODELPATH=argv[8]
    MODEL=argv[9]
    if argv[10]=="0":
    	ONLY15=False
    else:
    	ONLY15=True
    #conf = Configuration.Configuration()
    #conf.set_BODY_MODEL(argv[11])
    BODY_MODEL=eval(argv[11])
    NORMALIZATION=argv[12]
    if argv[13]=="0":
        KEYPOINT_RESTORATION=False
    else:
        KEYPOINT_RESTORATION=True
    NZ=int(argv[14])


except ValueError:
    print("Wrong arguments. Expecting two paths.")



####### INITIAL WARNINGS ########
if not DATASET_TEST=="dynamicData/H36Mtest":
    print(colors.CRED + "DATASET_TEST=" + str(DATASET_TEST) + colors.CEND)
else:
    print(colors.CGREEN + "DATASET_TEST=" + str(DATASET_TEST) + colors.CEND)

if not NORMALIZATION=="center_scale":
    print(colors.CRED + "NORMALIZATION=" + str(NORMALIZATION) + colors.CEND)
else:
    print(colors.CGREEN + "NORMALIZATION=" + str(NORMALIZATION) + colors.CEND)
if not KEYPOINT_RESTORATION:
    print(colors.CRED + "KEYPOINT_RESTORATION=" + str(KEYPOINT_RESTORATION) + colors.CEND)
else:
    print(colors.CGREEN + "KEYPOINT_RESTORATION=" + str(KEYPOINT_RESTORATION) + colors.CEND)
if MODEL=="models_mirror":
    print(colors.CRED + "MODEL=" + str(MODEL) + colors.CEND)
else:
    print(colors.CGREEN + "MODEL=" + str(MODEL) + colors.CEND)
if NZ!=100:
    print(colors.CRED + "NZ=" + str(NZ) + colors.CEND)
else:
    print(colors.CGREEN + "NZ=" + str(NZ) + colors.CEND)

###########

models = importlib.import_module(MODEL)

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

numJoints = len(BODY_MODEL.POSE_BODY_25_BODY_PARTS)  #15

# Size of z latent vector (i.e. size of generator input)
nz = NZ

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

theModels = models.Models(ngpu, numJoints, nz, KEYPOINT_RESTORATION, device = torch.device("cpu"))

models.load(theModels, MODELPATH)

def testMany(theModels, keypointsPath, imagesPath, outputPath, outputSubpath, imageExtension, saveImages=True):
    print('testMany('+keypointsPath+')')
    pathlib.Path(outputPath+outputSubpath+"/images").mkdir(parents=True, exist_ok=True)
    pathlib.Path(outputPath+outputSubpath+"/keypoints").mkdir(parents=True, exist_ok=True)
    
    batch_of_one_keypoints_cropped = []
    batch_of_one_confidence_values = []
    batch_scaleFactor = []
    batch_x_displacement = []
    batch_y_displacement = []
    batch_filenames = []
    batch_of_one_keypoints_cropped25 = []
    jsonFiles = [f for f in listdir(keypointsPath) if isfile(join(keypointsPath, f))]
    n = 0
    for filename in jsonFiles:
        #print('Testing '+filename)
        try:
            if ONLY15:
                only15joints=True
            else:
                only15joints=False
            #keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(join(keypointsPath, filename), BODY_MODEL, only15joints, NORMALIZATION, mean, std)

            keypoints_cropped = openPoseUtils.json2Keypoints(join(keypointsPath, filename), only15joints)
            confidence_values = openPoseUtils.getConfidence(keypoints_cropped)
            keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalizeV2(keypoints_cropped, BODY_MODEL, NORMALIZATION, False, mean, std)

            #print("keypoints_cropped.shape:", keypoints_cropped.shape)

            #We have worked with jut 15 keypoints, need to restore the other 10
            keypoints_cropped25 = openPoseUtils.json2Keypoints(join(keypointsPath, filename), False)
            sys.stdout.write(".")
            batch_of_one_keypoints_cropped25.append(keypoints_cropped25)

            #keypoints_cropped, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
            #keypoints_cropped = [item for sublist in keypoints_cropped for item in sublist]
            #keypoints_cropped = [float(k) for k in keypoints_cropped]
            keypoints_cropped = torch.tensor(keypoints_cropped)
            confidence_values = torch.tensor(confidence_values)
            #keypoints_cropped = keypoints_cropped.flatten()
            batch_of_one_keypoints_cropped.append(keypoints_cropped)
            batch_of_one_confidence_values.append(confidence_values)
            batch_scaleFactor.append(scaleFactor)
            batch_x_displacement.append(x_displacement)
            batch_y_displacement.append(y_displacement)
            batch_filenames.append(filename)
            n += 1
        except Exception as e: 
            print('Skipping '+filename+": "+str(e))
            #print(e)
            traceback.print_exc()
            #pass
            #print('Skipping '+filename)

    sys.stdout.write("\n")
	
    batch_of_one_keypoints_cropped = torch.stack(batch_of_one_keypoints_cropped)
    batch_of_one_confidence_values = torch.stack(batch_of_one_confidence_values)
    fixed_noise_N = torch.randn(n, nz)

    #Perform inference over ALL the poses
    fakeReshapedAsKeypoints = models.inference(theModels, n, fixed_noise_N, numJoints, batch_of_one_keypoints_cropped, batch_of_one_confidence_values)
    
    #Process inference results
    for idx in range(len(fakeReshapedAsKeypoints)):
        #Write keypoints to a file
        fakeKeypointsOneImage = fakeReshapedAsKeypoints[idx]
        
        fakeKeypointsCroppedOneImageIntRescaled = openPoseUtils.denormalizeV2(fakeKeypointsOneImage, batch_scaleFactor[idx], batch_x_displacement[idx], batch_y_displacement[idx], NORMALIZATION, keepConfidence=False, mean=mean, std=std, norm=None)
        
        #FIX
		#We have worked with jut 15 keypoints, need to restore the other 10
        #print("fakeKeypointsCroppedOneImageIntRescaled PRE", fakeKeypointsCroppedOneImageIntRescaled)
        keypoints_cropped25 = batch_of_one_keypoints_cropped25[idx]
        for i in range(len(fakeKeypointsCroppedOneImageIntRescaled), len(batch_of_one_keypoints_cropped25[idx])):
        	fakeKeypointsCroppedOneImageIntRescaled.append(keypoints_cropped25[i])
        #print("fakeKeypointsCroppedOneImageIntRescaled POST", fakeKeypointsCroppedOneImageIntRescaled)
        #till here
        if idx == 0:
            print("before writting to disk: ", fakeKeypointsCroppedOneImageIntRescaled[14])

        openPoseUtils.keypoints2json(fakeKeypointsCroppedOneImageIntRescaled, outputPath+outputSubpath+"/keypoints/"+batch_filenames[idx])
        #print("Wrote pose in %s in with %d keypoints" % (batch_filenames[idx], len(fakeKeypointsCroppedOneImageIntRescaled)))
    
        #Draw over image file
        #fileName = batch_filenames[idx]
        #indexUnderscore = fileName.find('_')
        #json_file_without_extension = fileName[:indexUnderscore] 

        #Prpeare image file path.
        #Remove "_keypoints" and anyghing else till the .
        #WANRING: CHARADE and TEST keypoints have slightly different names
        #CHARADE has other "_" that have to be preserved
             
        json_file_without_extension = os.path.splitext(batch_filenames[idx])[0]

        indexFromRemove = json_file_without_extension.find('_keypoints')
        json_file_without_extension = json_file_without_extension[:indexFromRemove]

        #json_file_without_extension = json_file_without_extension.replace('_keypoints', '')
        originalImagePath = join(imagesPath, json_file_without_extension+imageExtension)
        #print("originalImagePath="+originalImagePath)
            
        if saveImages:
            imgWithKyepoints = pytorchUtils.cv2ReadFile(originalImagePath)
            poseUtils.draw_pose(imgWithKyepoints, fakeKeypointsCroppedOneImageIntRescaled, -1, BODY_MODEL.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
            try:
                cv2.imwrite(outputPath+outputSubpath+"/images/"+json_file_without_extension+".jpg", imgWithKyepoints)
                #print("written data/output/Test/"+json_file_without_extension+".jpg")
            except:
                print("WARNING: Cannot find "+originalImagePath)  
            #shutil.copyfile(join("/Users/rtous/DockerVolume/openpose/data/result", json_file_without_extension+"_rendered.png"), join("data/output/Test/"+json_file_without_extension+"_img_cropped_openpose.png"))
        else:
            WIDTH = 500
            HEIGHT = 500
            blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)
            #poseUtils.draw_pose(blank_image, fakeKeypointsCroppedOneImageIntRescaled, -1, BODY_MODEL.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
            poseUtils.draw_pose_scaled_centered(blank_image, np.array(fakeKeypointsCroppedOneImageIntRescaled), -1, BODY_MODEL.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 500/WIDTH, WIDTH/2, HEIGHT/2, 8)           
            cv2.imwrite(outputPath+outputSubpath+"/images/"+json_file_without_extension+".jpg", blank_image)
    print('testMany() finished. Output written to '+outputPath+outputSubpath)        

#old and should be fixed but not used in the workflow
def testImage(netG, outputPath, imagePath, keypointsPath):
    #Test over the test image
    keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(keypointsPath, BODY_MODEL)
    print("scaleFactor=",scaleFactor)
    keypoints_cropped, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
    keypoints_cropped = [item for sublist in keypoints_cropped for item in sublist]
    keypoints_cropped = [float(k) for k in keypoints_cropped]
    keypoints_cropped = torch.tensor(keypoints_cropped)
    confidence_values = torch.tensor(confidence_values)
    keypoints_cropped = keypoints_cropped.flatten()
    print("keypoints_cropped.shape = ", keypoints_cropped.shape)

    batch_of_one_keypoints_cropped = np.reshape(keypoints_cropped, (1, numJoints*2))
    batch_of_one_confidence_values = np.reshape(confidence_values, (1, numJoints))
    fixed_noise_one = torch.randn(1, nz, device=device)

    batch_of_one_keypoints_cropped = batch_of_one_keypoints_cropped.to(device)
    fixed_noise_one = fixed_noise_one.to(device)

    netG.eval()
    fake = netG(batch_of_one_keypoints_cropped, fixed_noise_one).detach().cpu()
    if KEYPOINT_RESTORATION:
        fake = restoreOriginalKeypoints(fake, batch_of_one_keypoints_cropped, batch_of_one_confidence_values)
    netG.train()
    fakeReshapedAsKeypoints = np.reshape(fake, (1, numJoints, 2))
    fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()

    fakeKeypointsOneImage = fakeReshapedAsKeypoints[0]
    fakeKeypointsOneImage, dummy, dummy, dummy = openPoseUtils.normalize(fakeKeypointsOneImage)
    #fakeKeypointsOneImageInt = poseUtils.keypointsToInteger(fakeKeypointsOneImage)

    fakeKeypointsCroppedOneImageIntRescaled = openPoseUtils.denormalize(fakeKeypointsOneImage, scaleFactor, x_displacement, y_displacement)
    #imgWithKyepoints = np.zeros((500, 500, 3), np.uint8)
    imgWithKyepoints = cv2.imread(imagePath)
    poseUtils.draw_pose(imgWithKyepoints, fakeKeypointsCroppedOneImageIntRescaled, -1, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
    cv2.imwrite(outputPath+"/test_keypoints.jpg", imgWithKyepoints)

	
#CHARADE DATASET
testMany(theModels, DATASET_CHARADE, DATASET_CHARADE_IMAGES, OUTPUTPATH, "/CHARADE", ".png", True)

#TEST DATASET
testMany(theModels, DATASET_TEST, DATASET_TEST_IMAGES, OUTPUTPATH, "/TEST", ".jpg", False)

#testMany(netG, DATASET_TEST, DATASET_TEST_IMAGES, OUTPUTPATH, "/TEST", ".jpg")

        