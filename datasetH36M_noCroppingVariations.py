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
import h36mIterator_tiny as h36mIterator #DEBUG import h36mIterator
import BodyModelOPENPOSE15

class JsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, inputpath_cropped, inputpath_original, bodyModel):
        self.inputpath_cropped = inputpath_cropped
        self.inputpath_original = inputpath_original
        self.bodyModel = bodyModel
        #self.count = countFiles(self.inputpath_cropped, ".json")
        #self.jsonFiles = [f for f in listdir(self.inputpath_cropped) if isfile(join(self.inputpath_cropped, f)) and f.endswith("json") ]
        
        #scandir does not need to read the entire file list first
        #self.scandirIterator = os.scandir(self.inputpath_cropped)

    def __iter__(self):
        #jsonFiles = [f for f in listdir(self.inputpath) if isfile(join(self.inputpath, f)) and f.endswith("json") ]
        #Important, the scandir iterator needs to be created each time
        #buffer = []
        #self.scandirIterator = os.scandir(self.inputpath_original)
        self.scandirIterator = h36mIterator.iterator(self.inputpath_original)
        buffer_originals = []
        for keypoints_original in self.scandirIterator:
            #We fill first a buffer of originals
            buffer_originals.append(keypoints_original)
            len_buffer_originals = len(buffer_originals)
            if len_buffer_originals == 65536:#65536
                break
                
        #Once the iterator finishes or the buffer is filled, we shuffle and obtain variations
        print("shuffle buffer original full: ", len_buffer_originals)
        print("sorting buffer originals...")
        random.shuffle(buffer_originals)
        print("yielding originals...")

        for buffered_keypoints_original in buffer_originals:
            keypoints_original_norm, dummy, dummy, dummy = openPoseUtils.normalize(buffered_keypoints_original, BodyModelOPENPOSE15, keepConfidence=False)
            keypoints_original_norm_noconfidence_flat = [item for sublist in keypoints_original_norm for item in sublist]
            keypoints_original_flat = torch.tensor(keypoints_original_norm_noconfidence_flat)
            
            
            keypoints_cropped_norm, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalize(buffered_keypoints_original, BodyModelOPENPOSE15, keepConfidence=True)              
            #keypoints_cropped_norm, dummy, dummy, dummy = openPoseUtils.normalize(keypoints_cropped, keepConfidence=True)             
            #dummy, dummy, dummy = openPoseUtils.normalize(keypoints_cropped, keepConfidence=True)             
            
            keypoints_croppedNoConf, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped_norm)
            keypoints_croppedFlat = [item for sublist in keypoints_croppedNoConf for item in sublist]
            keypoints_croppedFlatFloat = [float(k) for k in keypoints_croppedFlat]
            keypoints_croppedFlatFloatTensor = torch.tensor(keypoints_croppedFlatFloat)
            
            #The confidence_values of the cropped skeletons signal which bones to fake and which to keep
            confidence_values = torch.tensor(confidence_values)#Here I repeat the originals twice because
            #some training expect the variation to come first
            yield keypoints_croppedFlatFloatTensor, keypoints_original_flat, confidence_values, scaleFactor, x_displacement, y_displacement, "unknown file"
    
        self.scandirIterator.close()
        print("Closed h36mIterator.")
            

    #def __len__(self):
    #    return self.count
    #    return len(self.jsonFiles)
'''
Here I change the nose to be a simple batchsize x 100 tensor.
I order to input this into the deconvolution I did within the forward:
inputReshaped = input.view(b_size, nz, 1, 1)
return self.main(inputReshaped)
'''
