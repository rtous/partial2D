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

import poseUtils
import cv2
import traceback
import shutil
import sys
from torch.utils.tensorboard import SummaryWriter

#import models
import dataset
import datasetBasic
#import datasetBACKUP
import datasetH36M
#import datasetH36M_mem
import time
#import Configuration
import openPoseUtils
import importlib

sys.set_int_max_str_digits(0)



def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file in dataloader:
        # Mean over batch, height and width, but not over the channels
        data = batch_of_keypoints_original
        channels_sum += torch.mean(data)#,dim=[0,1]
        channels_squared_sum += torch.mean(data**2)
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean.numpy(), std.numpy()

def split_dataset(anIterableDataset, TRAINSPLIT, batch_size, workers):
    buffer_variations = []
    for batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file in anIterableDataset:
        buffer_variations.append((batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file))
    train_set_size = int(len(buffer_variations) * TRAINSPLIT)
    valid_set_size = len(buffer_variations) - train_set_size
    train_set, valid_set = torch.utils.data.random_split(buffer_variations, [train_set_size, valid_set_size])
    dataloader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=workers)
    dataloaderValidation  = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=workers)
    return dataloader, dataloaderValidation

def writeRunInfoFile(run_info_json, run_info_file_path, OUTPUTPATH):
    run_info_file = open(OUTPUTPATH+run_info_file_path, 'w')
    json.dump(run_info_json, run_info_file)
    run_info_file.flush()
    run_info_file.close()

def drawGrid(fakeReshapedAsKeypoints, originalReshapedAsKeypoints, croppedReshapedAsKeypoints, OUTPUTPATH, batch_size, BODY_MODEL, scaleFactor, x_displacement, y_displacement, batch_of_json_file, NORMALIZATION, mean, std):
    NUM_ROWS = int(batch_size/16)
    NUM_COLS = int(batch_size/16)
    WIDTH = 64
    HEIGHT = 64
    imagesCropped = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
    images = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
    imagesOriginal = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
            

    ####### DRAW DEBUG POSES FOR THE FIRST 64 IMAGES
    print("Draw debug poses for the batch")
    #print("denormalizing with conf.norm=", conf.norm)
    for idx in range(NUM_ROWS*NUM_COLS):
        blank_imageOriginal = np.zeros((WIDTH,HEIGHT,3), np.uint8)
        blank_imageCropped = np.zeros((WIDTH,HEIGHT,3), np.uint8)
        blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)
        originalReshapedAsKeypointsOneImage = originalReshapedAsKeypoints[idx]
        fakeKeypointsCroppedOneImage = croppedReshapedAsKeypoints[idx]
        fakeKeypointsOneImage = fakeReshapedAsKeypoints[idx]

        scaleFactorOneImage = scaleFactor[idx]
        x_displacementOneImage = x_displacement[idx]
        y_displacementOneImage = y_displacement[idx]

        json_file = batch_of_json_file[idx]
        
        originalReshapedAsKeypointsOneImageInt = originalReshapedAsKeypointsOneImage
        fakeKeypointsCroppedOneImageInt = fakeKeypointsCroppedOneImage
        fakeKeypointsOneImageInt = fakeKeypointsOneImage
        
        
        #Draw result over the original image
        
        #Denormalize
        #Cannot denormalize originals as we don't keep the normalization info
        #Denormalize input
        fakeKeypointsCroppedOneImageIntDenormalized = openPoseUtils.denormalizeV2(fakeKeypointsCroppedOneImageInt, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage, NORMALIZATION, keepConfidence=False, mean=mean, std=std)#conf.norm)
        #Denormalize output
        fakeKeypointsOneImageIntDenormalized = openPoseUtils.denormalizeV2(fakeKeypointsOneImageInt, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage, NORMALIZATION, keepConfidence=False, mean=mean, std=std)#conf.norm)
        #print("fakeKeypointsCroppedOneImageIntDenormalized:", fakeKeypointsCroppedOneImageIntDenormalized)
        #print("fakeKeypointsOneImageIntDenormalized:",fakeKeypointsOneImageIntDenormalized)

        #test
        #originalReshapedAsKeypointsOneImageInt = openPoseUtils.denormalizeV2(originalReshapedAsKeypointsOneImageInt, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage, NORMALIZATION, keepConfidence=False, mean=mean, std=std)#conf.norm)


        #fakeKeypointsCroppedOneImageIntRescaledNP = poseUtils.keypoints2Numpy(fakeKeypointsCroppedOneImageIntRescaled)
        #fakeKeypointsCroppedOneImageIntRescaledNP = poseUtils.scale(fakeKeypointsCroppedOneImageIntRescaledNP, 0.01)


        #If we want to save the .json files of the batch
        #openPoseUtils.keypoints2json(fakeKeypointsOneImageInt, OUTPUTPATH+"/"+f"{idx:02d}"+"_img_keypoints.json")
        
        #json_file_without_extension = os.path.splitext(json_file)[0]
        #json_file_without_extension = json_file_without_extension.replace('_keypoints', '')
        
        #Draw the (still normalized) results for DEBUG  
        #NOTE: the original poses are centered at (0.5, 0.5)
        #They are re-escaled and centered at the mid hip for drawing
        #originalReshapedAsKeypointsOneImageInt
        #fakeKeypointsCroppedOneImageInt
        #fakeKeypointsOneImageInt (with restoreOriginalKeypoints)
        try:
            #poseUtils.draw_pose(blank_imageOriginal, originalReshapedAsKeypointsOneImageInt, -1, conf.bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
            #poseUtils.draw_pose(blank_imageCropped, fakeKeypointsCroppedOneImageInt, -1, conf.bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
            #poseUtils.draw_pose(blank_image, fakeKeypointsOneImageInt, -1, conf.bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
            #print("fakeKeypointsCroppedOneImageIntDenormalized=",fakeKeypointsCroppedOneImageIntDenormalized)
            
            #poseUtils.draw_pose_scaled_centered(blank_imageOriginal, np.array(originalReshapedAsKeypointsOneImageInt), -1, BODY_MODEL.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 1/(WIDTH/4), WIDTH/2, HEIGHT/2, 8)           
            poseUtils.draw_pose_scaled_centered(blank_imageCropped, np.array(fakeKeypointsCroppedOneImageIntDenormalized), -1, BODY_MODEL.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 8, WIDTH/2, HEIGHT/2, 8)
            poseUtils.draw_pose_scaled_centered(blank_image, np.array(fakeKeypointsOneImageIntDenormalized), -1, BODY_MODEL.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 8, WIDTH/2, HEIGHT/2, 8)
            targetFilePathCropped = OUTPUTPATH+"/debug_input"+str(idx)+".jpg"
            targetFilePath = OUTPUTPATH+"/debug"+str(idx)+".jpg"
            #cv2.imwrite(targetFilePath, blank_image)
            imagesOriginal[int(idx/NUM_COLS)][int(idx%NUM_COLS)] = blank_imageOriginal
            imagesCropped[int(idx/NUM_COLS)][int(idx%NUM_COLS)] = blank_imageCropped
            images[int(idx/NUM_COLS)][int(idx%NUM_COLS)] = blank_image
        except Exception:
            print("WARNING: Cannot draw keypoints ", fakeKeypointsOneImageInt)
            traceback.print_exc()
    try:
        #print("Assigning: images[int("+str(idx)+"/NUM_COLS)][int("+str(idx)+"%NUM_COLS)]")
        total_imageOriginal = poseUtils.concat_tile(imagesOriginal)
        total_imageCropped = poseUtils.concat_tile(imagesCropped)
        total_image = poseUtils.concat_tile(images)  
        targetFilePathOriginal = OUTPUTPATH+"/debug_input_original.jpg"
        targetFilePathCropped = OUTPUTPATH+"/debug_input_cropped.jpg"
        targetFilePath = OUTPUTPATH+"/debug_output.jpg"
        print("Writting into ", targetFilePathOriginal)
        cv2.imwrite(targetFilePathCropped, total_imageCropped)
        cv2.imwrite(targetFilePath, total_image)
        cv2.imwrite(targetFilePathOriginal, total_imageOriginal)
    except Exception:
        print("WARNING: Cannot draw tile ")
        traceback.print_exc()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

