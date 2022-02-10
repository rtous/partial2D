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
import models
import random

class JsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, inputpath_cropped, inputpath_original):
        self.inputpath_cropped = inputpath_cropped
        self.inputpath_original = inputpath_original
        #self.count = countFiles(self.inputpath_cropped, ".json")
        #self.jsonFiles = [f for f in listdir(self.inputpath_cropped) if isfile(join(self.inputpath_cropped, f)) and f.endswith("json") ]
        
        #scandir does not need to read the entire file list first
        #self.scandirIterator = os.scandir(self.inputpath_cropped)

    def __iter__(self):
        #jsonFiles = [f for f in listdir(self.inputpath) if isfile(join(self.inputpath, f)) and f.endswith("json") ]
        #Important, the scandir iterator needs to be created each time
        buffer = []
        self.scandirIterator = os.scandir(self.inputpath_original)
        for item in self.scandirIterator:
            json_file = str(item.name)
            if json_file.endswith(".json"):
                try:
                    keypoints_original, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(join(self.inputpath_original, json_file))
                    #confidence_values = openPoseUtils.getConfidence(keypoints_original)
                    keypoints_original_noconfidence, dummy = openPoseUtils.removeConfidence(keypoints_original)
                    
                    keypoints_original_flat = [item for sublist in keypoints_original_noconfidence for item in sublist]
                    #print("keypoints_original_flat:")
                    #print(keypoints_original_flat)
                    keypoints_original_flat = torch.tensor(keypoints_original_flat)
                    keypoints_original_flat = keypoints_original_flat.flatten()
                    #print("keypoints_original_flatten:")
                    #print(keypoints_original_flat)
                    '''
                    variations = openPoseUtils.crop(keypoints_original)
                    keypoints_cropped, dummy = openPoseUtils.removeConfidence(variations[0])
                    keypoints_cropped = [item for sublist in keypoints_cropped for item in sublist]
                    keypoints_cropped = [float(k) for k in keypoints_cropped]
                    keypoints_cropped = torch.tensor(keypoints_cropped)
                    keypoints_cropped = keypoints_cropped.flatten()
                    '''
                    variations = openPoseUtils.crop(keypoints_original)
                    #print("variations = ", len(variations))
                    for v_idx, keypoints_cropped in enumerate(variations):
                        #print(keypoints_cropped)
                        keypoints_croppedNoConf, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
                        keypoints_croppedFlat = [item for sublist in keypoints_croppedNoConf for item in sublist]
                        keypoints_croppedFlatFloat = [float(k) for k in keypoints_croppedFlat]
                        keypoints_croppedFlatFloatTensor = torch.tensor(keypoints_croppedFlatFloat)
                        #keypoints_cropped = keypoints_cropped.flatten()
                        #print("variation:", keypoints_croppedFlatFloatTensor)
                    
                        #print("yielding: keypoints_cropped.shape="+str(keypoints_cropped.shape)+" keypoints_original_flat.shape="+str(keypoints_original_flat.shape))   
                        
                        #INFO: The confidence_values of the cropped skeletons signal which bones to fake and which to keep
                        confidence_values = torch.tensor(confidence_values)
                        buffer.append((keypoints_croppedFlatFloatTensor, keypoints_original_flat, confidence_values, scaleFactor, x_displacement, y_displacement, json_file))
                        if len(buffer) == 2048:
                            random.shuffle(buffer)
                            for tup in buffer:
                                #print("yield with num of null bones: ", openPoseUtils.numOfNullJoints(tup[0]))
                                #print("yield: ", tup[0])

                                yield tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6]

                            buffer = []
                        #yield keypoints_cropped, keypoints_original_flat, confidence_values, scaleFactor, x_displacement, y_displacement, json_file
                except ValueError as ve:
                    print(ve)
                #except OSError as oe:
                #   print(oe)
                except Exception as e:
                    print("WARNING: Error reading ", json_file)
                    #print(e)
                    traceback.print_exc()
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
