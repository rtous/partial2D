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
import Configuration
#import models

CRED = '\033[91m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CBOLD     = '\33[1m'
CEND = '\033[0m'

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
    conf = Configuration.Configuration()
    conf.set_BODY_MODEL(argv[11])
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
    print(CRED + "DATASET_TEST=" + str(DATASET_TEST) + CEND)
else:
    print(CGREEN + "DATASET_TEST=" + str(DATASET_TEST) + CEND)

if not NORMALIZATION=="center_scale":
    print(CRED + "NORMALIZATION=" + str(NORMALIZATION) + CEND)
else:
    print(CGREEN + "NORMALIZATION=" + str(NORMALIZATION) + CEND)
if not KEYPOINT_RESTORATION:
    print(CRED + "KEYPOINT_RESTORATION=" + str(KEYPOINT_RESTORATION) + CEND)
else:
    print(CGREEN + "KEYPOINT_RESTORATION=" + str(KEYPOINT_RESTORATION) + CEND)
if MODEL=="models_mirror":
    print(CRED + "MODEL=" + str(MODEL) + CEND)
else:
    print(CGREEN + "MODEL=" + str(MODEL) + CEND)
if NZ!=100:
    print(CRED + "NZ=" + str(NZ) + CEND)
else:
    print(CGREEN + "NZ=" + str(NZ) + CEND)

###########
models = importlib.import_module(MODEL)

pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 
#pathlib.Path(OUTPUTPATH+"/Test/").mkdir(parents=True, exist_ok=True) 
#pathlib.Path(OUTPUTPATH+"/Test/keypoints").mkdir(parents=True, exist_ok=True) 
#pathlib.Path(OUTPUTPATH+"/Test/images").mkdir(parents=True, exist_ok=True) 


batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
#numJoints = 15#25
numJoints = len(conf.bodyModel.POSE_BODY_25_BODY_PARTS)  #15
image_size = numJoints*2

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = NZ

# Size of feature maps in generator
#ngf = 64
ngf = 16

# Size of feature maps in discriminator
#ndf = 64
ndf = 16

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0

print("numJoints=", str(numJoints))
netG = models.Generator(ngpu, numJoints, nz)
netG.load_state_dict(torch.load(MODELPATH))
#model.eval()



def testMany(netG, keypointsPath, imagesPath, outputPath, outputSubpath, imageExtension, saveImages=True):
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
            keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(join(keypointsPath, filename), conf.bodyModel, only15joints, NORMALIZATION, mean, std)
            #print("keypoints_cropped[0]=", keypoints_cropped[0])

            #We have worked with jut 15 keypoints, need to restore the other 10
            keypoints_cropped25 = openPoseUtils.json2Keypoints(join(keypointsPath, filename), False)
            #print("keypoints_cropped25[0]=", keypoints_cropped25[0])
            #print("Read %d keypoints." % (len(keypoints_cropped25)))
            sys.stdout.write(".")
            batch_of_one_keypoints_cropped25.append(keypoints_cropped25)


            #print("normalized keypoints_cropped=", keypoints_cropped)
            #print("obtained scaleFactor=",scaleFactor)
            keypoints_cropped, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
            keypoints_cropped = [item for sublist in keypoints_cropped for item in sublist]
            keypoints_cropped = [float(k) for k in keypoints_cropped]
            keypoints_cropped = torch.tensor(keypoints_cropped)
            confidence_values = torch.tensor(confidence_values)
            keypoints_cropped = keypoints_cropped.flatten()
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
	#batch_of_one_keypoints_cropped = torch.tensor(batch_of_one_keypoints_cropped)
	#batch_of_one_confidence_values = torch.tensor(batch_of_one_confidence_values)
    batch_of_one_keypoints_cropped = torch.stack(batch_of_one_keypoints_cropped)
    batch_of_one_confidence_values = torch.stack(batch_of_one_confidence_values)
    #fixed_noise_N = torch.randn(n, nz, device=device)
    fixed_noise_N = torch.randn(n, nz)


    #batch_of_one_keypoints_cropped = batch_of_one_keypoints_cropped.to(device)
    #fixed_noise_N = fixed_noise_N.to(device)

    print("batch_of_one_keypoints_cropped.shape:")
    print(batch_of_one_keypoints_cropped.shape)

    netG.eval()
    #print("input[0]=", batch_of_one_keypoints_cropped[0])
    #fake = netG(batch_of_one_keypoints_cropped, fixed_noise_N).detach().cpu()
    

    #latentZ = netG.encode(batch_of_one_keypoints_cropped).detach().cpu()
    #fake = netG.decode(latentZ).detach().cpu()

    fake = netG.decode(fixed_noise_N).detach().cpu()
    
    #print("fake[0]=", fake[0])

    #fake = batch_of_one_keypoints_cropped # DEBUG DOING NOTHING

    #print("cropped before restoring:", batch_of_one_keypoints_cropped[0])
    #print("fake before restoring:", fake[0])
    #print("fake before restoring have len %d" % len(fake[0]))
    #fakeDeflatten = poseUtils.deflatten(fake[0], False)
    #print("fake before restoring:", poseUtils.deflatten(fake[0], False))
    #print("batch_of_one_confidence_values:", batch_of_one_confidence_values[0])
    



    if KEYPOINT_RESTORATION:
        fake = models.restoreOriginalKeypoints(fake, batch_of_one_keypoints_cropped, batch_of_one_confidence_values)
    
    

    #fakeRestoredDeflatten = poseUtils.deflatten(fake[0], False)
    
    #print("after restoring:", fakeRestoredDeflatten)
    

    ######## DEBUG RESTORATION PROCESS 
    '''
    for idx in range(len(fake)):
        print(batch_filenames[idx])
        fakeDeflatten = poseUtils.deflatten(fake[idx], False)
        fakeRestoredDeflatten = poseUtils.deflatten(fakeRestored[idx], False)
        batch_of_one_keypoints_croppedDeflatten = poseUtils.deflatten(batch_of_one_keypoints_cropped[idx], False)
        for i, k in enumerate(fakeDeflatten):
            #print("[%d,%d]" % (fakeDeflatten[i][0], fakeDeflatten[i][1]))
            if batch_of_one_confidence_values[idx][i] > 0.1 and ((fakeDeflatten[i][0] == fakeRestoredDeflatten[i][0]) and (fakeDeflatten[i][0] == fakeRestoredDeflatten[i][0])):
                print("REST(%d)[%f,%f,\033[94m%f\033[0m]->[%f,%f]->\033[91m[%f,%f]\033[0m" % (i, batch_of_one_keypoints_croppedDeflatten[i][0], batch_of_one_keypoints_croppedDeflatten[i][1], batch_of_one_confidence_values[idx][i], fakeDeflatten[i][0], fakeDeflatten[i][1], fakeRestoredDeflatten[i][0], fakeRestoredDeflatten[i][1]))
            elif batch_of_one_confidence_values[idx][i] > 0.1 :
                print("REST(%d)[%f,%f,\033[94m%f\033[0m]->[%f,%f]->\033[92m[%f,%f]\033[0m" % (i, batch_of_one_keypoints_croppedDeflatten[i][0], batch_of_one_keypoints_croppedDeflatten[i][1], batch_of_one_confidence_values[idx][i], fakeDeflatten[i][0], fakeDeflatten[i][1], fakeRestoredDeflatten[i][0], fakeRestoredDeflatten[i][1]))
            elif batch_of_one_confidence_values[idx][i] <= 0.1 and ((fakeDeflatten[i][0] != fakeRestoredDeflatten[i][0]) or (fakeDeflatten[i][0] != fakeRestoredDeflatten[i][0])):
                print("GUES(%d)[%f,%f,\033[93m%f\033[0m]->[%f,%f]->\033[91m[%f,%f]\033[0m" % (i, batch_of_one_keypoints_croppedDeflatten[i][0], batch_of_one_keypoints_croppedDeflatten[i][1], batch_of_one_confidence_values[idx][i], fakeDeflatten[i][0], fakeDeflatten[i][1], fakeRestoredDeflatten[i][0], fakeRestoredDeflatten[i][1]))
            elif batch_of_one_confidence_values[idx][i] <= 0.1:
                print("GUES(%d)[%f,%f,\033[93m%f\033[0m]->[%f,%f]->\033[92m[%f,%f]\033[0m" % (i, batch_of_one_keypoints_croppedDeflatten[i][0], batch_of_one_keypoints_croppedDeflatten[i][1], batch_of_one_confidence_values[idx][i], fakeDeflatten[i][0], fakeDeflatten[i][1], fakeRestoredDeflatten[i][0], fakeRestoredDeflatten[i][1]))
            else:
                print("????(%d)[%f,%f,%f]->[%f,%f]\033[0m" % (i, batch_of_one_keypoints_croppedDeflatten[i][0], batch_of_one_keypoints_croppedDeflatten[i][1], batch_of_one_confidence_values[idx][i], fakeDeflatten[i][0], fakeDeflatten[i][1], fakeRestoredDeflatten[i][0], fakeRestoredDeflatten[i][1]))
    '''
    ########
    
    netG.train()
    fakeReshapedAsKeypoints = np.reshape(fake, (n, numJoints, 2))
    fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()


    for idx in range(len(fakeReshapedAsKeypoints)):
        #Write keypoints to a file
        fakeKeypointsOneImage = fakeReshapedAsKeypoints[idx]
        
        #???? this was here before
        #fakeKeypointsOneImage, dummy, dummy, dummy = openPoseUtils.normalize(fakeKeypointsOneImage)
        #print("before denormalize: ", fakeKeypointsOneImage[14])
        fakeKeypointsCroppedOneImageIntRescaled = openPoseUtils.denormalizeV2(fakeKeypointsOneImage, batch_scaleFactor[idx], batch_x_displacement[idx], batch_y_displacement[idx], NORMALIZATION, keepConfidence=False, mean=mean, std=std, norm=None)
        #print("after denormalize: ", fakeKeypointsCroppedOneImageIntRescaled[14])
        #if idx == 0:
        #    print("writing: ", batch_filenames[idx])
        #    print("written: ", fakeKeypointsCroppedOneImageIntRescaled)
        
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
            poseUtils.draw_pose(imgWithKyepoints, fakeKeypointsCroppedOneImageIntRescaled, -1, conf.bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
            try:
                cv2.imwrite(outputPath+outputSubpath+"/images/"+json_file_without_extension+".jpg", imgWithKyepoints)
                #print("written data/output/Test/"+json_file_without_extension+".jpg")
            except:
                print("WARNING: Cannot find "+originalImagePath)  
            #shutil.copyfile(join("/Users/rtous/DockerVolume/openpose/data/result", json_file_without_extension+"_rendered.png"), join("data/output/Test/"+json_file_without_extension+"_img_cropped_openpose.png"))
        else:
            blank_image = np.zeros((1000,1000,3), np.uint8)
            cv2.imwrite(outputPath+outputSubpath+"/images/"+json_file_without_extension+".jpg", blank_image)
    print('testMany() finished. Output written to '+outputPath+outputSubpath)        

def testImage(netG, outputPath, imagePath, keypointsPath):
    #Test over the test image
    keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(keypointsPath, conf.bodyModel)
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
testMany(netG, DATASET_CHARADE, DATASET_CHARADE_IMAGES, OUTPUTPATH, "/CHARADE", ".png")

#TEST DATASET
testMany(netG, DATASET_TEST, DATASET_TEST_IMAGES, OUTPUTPATH, "/TEST", ".jpg", False)

#testMany(netG, DATASET_TEST, DATASET_TEST_IMAGES, OUTPUTPATH, "/TEST", ".jpg")

        