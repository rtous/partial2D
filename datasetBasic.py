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
#import models
import random

class JsonDataset(torch.utils.data.IterableDataset):
    #def __init__(self, inputpath_cropped, inputpath_original, bodyModel):
    def __init__(self, inputpath_cropped, inputpath_original, bodyModel, croppedVariations = True, normalizationStrategy = "center-scale", mean = None, std = None, max_len_buffer_originals = None, DISCARDINCOMPLETEPOSES = False):
        self.inputpath_cropped = inputpath_cropped
        self.inputpath_original = inputpath_original
        self.bodyModel = bodyModel
        self.normalization = normalizationStrategy
        self.mean = mean
        self.std = std
        if bodyModel.numJoints == 15:
            self.only15joints = True
        else:
            self.only15joints = False
        #self.count = countFiles(self.inputpath_cropped, ".json")
        #self.jsonFiles = [f for f in listdir(self.inputpath_cropped) if isfile(join(self.inputpath_cropped, f)) and f.endswith("json") ]
        
        #scandir does not need to read the entire file list first
        #self.scandirIterator = os.scandir(self.inputpath_cropped)

    def __iter__(self):
        #jsonFiles = [f for f in listdir(self.inputpath) if isfile(join(self.inputpath, f)) and f.endswith("json") ]
        #Important, the scandir iterator needs to be created each time
        self.scandirIterator = os.scandir(self.inputpath_cropped)
        i = 0
        for item in self.scandirIterator:
            json_file = str(item.name)
            if json_file.endswith(".json"):
                try:
                    #keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(join(self.inputpath_cropped, json_file), self.bodyModel)
                    

                    keypoints_cropped = openPoseUtils.json2Keypoints(join(self.inputpath_cropped, json_file), self.only15joints)
                    confidence_values = openPoseUtils.getConfidence(keypoints_cropped)
                    keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalizeV2(keypoints_cropped, self.bodyModel, self.normalization, False, self.mean, self.std)

                    '''
                    keypoints_cropped, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
                    keypoints_cropped = [item for sublist in keypoints_cropped for item in sublist]
                    keypoints_cropped = [float(k) for k in keypoints_cropped]
                    keypoints_cropped = torch.tensor(keypoints_cropped)
                    keypoints_cropped = keypoints_cropped.flatten()
                    '''

                    #Read the file with the original keypoints
                    #They are normalized
                    #They are used to 1)   2) restore good keypoints in the result
                    indexUnderscore = json_file.find('_')
                    json_file = json_file[:indexUnderscore]+".json"#+"_keypoints.json"  
                    original_keypoints_path = join(self.inputpath_original, json_file)
                    if not os.path.isfile(original_keypoints_path):
                        print("FATAL ERROR: original keypoints path not found: "+original_keypoints_path)
                        sys.exit()

                    '''
                    keypoints_original, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(original_keypoints_path, self.bodyModel)
                    keypoints_original, dummy = openPoseUtils.removeConfidence(keypoints_original)
                    keypoints_original = [item for sublist in keypoints_original for item in sublist]
                    keypoints_original = [float(k) for k in keypoints_original]
                    keypoints_original = torch.tensor(keypoints_original)
                    keypoints_original = keypoints_original.flatten()
                    confidence_values = torch.tensor(confidence_values)
                    '''
                    keypoints_original = openPoseUtils.json2Keypoints(original_keypoints_path, self.only15joints)
                    confidence_values = openPoseUtils.getConfidence(keypoints_original)
                    keypoints_original, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalizeV2(keypoints_original, self.bodyModel, self.normalization, False, self.mean, self.std)


                    #print("confidence_values:")
                    #print(confidence_values)
                    print("WARNING: OK reading ", join(self.inputpath_cropped, json_file))
                    print("i=",i)
                    i=i+1
                    yield keypoints_cropped, keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, json_file

                except ValueError as ve:
                    print(ve)
                #except OSError as oe:
                #   print(oe)
                except Exception as e:
                    print("WARNING: Error reading ", json_file)
                    print(e)
                    #traceback.print_exc()
        self.scandirIterator.close()
        print("Closed scandirIterator.")
            

    #def __len__(self):
    #    return self.count
    #    return len(self.jsonFiles)
'''
Here I change the nose to be a simple batchsize x 100 tensor.
I order to input this into the deconvolution I did within the forward:
inputReshaped = input.view(b_size, nz, 1, 1)
return self.main(inputReshaped)
'''
