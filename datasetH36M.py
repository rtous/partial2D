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
#import h36mIterator_tiny as h36mIterator #DEBUG import h36mIterator
import h36mIterator
import BodyModelOPENPOSE15
#import Configuration

class JsonDataset(torch.utils.data.IterableDataset):
    
    def __init__(self, inputpath_cropped, inputpath_original, bodyModel, croppedVariations = True, normalizationStrategy = "center-scale", mean = None, std = None, max_len_buffer_originals = None, DISCARDINCOMPLETEPOSES = False):
        self.inputpath_cropped = inputpath_cropped
        self.inputpath_original = inputpath_original
        self.bodyModel = bodyModel
        self.croppedVariations = croppedVariations
        self.normalizationStrategy = normalizationStrategy
        self.mean = mean
        self.std = std
        self.max_len_buffer_originals = max_len_buffer_originals
        self.DISCARDINCOMPLETEPOSES = DISCARDINCOMPLETEPOSES

    def __iter__(self):
        #jsonFiles = [f for f in listdir(self.inputpath) if isfile(join(self.inputpath, f)) and f.endswith("json") ]
        #Important, the scandir iterator needs to be created each time
        #buffer = []
        #self.scandirIterator = os.scandir(self.inputpath_original)
        buffer_originals = []
        buffer_variations = []
        self.scandirIterator = h36mIterator.iterator(self.inputpath_original, self.DISCARDINCOMPLETEPOSES)
        print("self.max_len_buffer_originals=", self.max_len_buffer_originals)
        for keypoints_original in self.scandirIterator:
            #We fill first a buffer of originals
            buffer_originals.append(keypoints_original)
            len_buffer_originals = len(buffer_originals)
            if len_buffer_originals == self.max_len_buffer_originals:
                print("self.max_len_buffer_originals=", self.max_len_buffer_originals)
                break
        
        print("computing norm...")
        #norm = self.computeNorm(buffer_originals)
        #self.conf.norm = norm
        #norm = 1
        #print("norm=", norm)
        #Once the iterator finishes or the buffer is filled, we shuffle and obtain variations
        print("shuffle buffer original full: ", len_buffer_originals)
        print("sorting buffer originals...")
        random.shuffle(buffer_originals)
        print("generating variations...")

        for buffered_keypoints_original in buffer_originals:
            try:
                keypoints_original_norm, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalizeV2(buffered_keypoints_original, BodyModelOPENPOSE15, self.normalizationStrategy, keepConfidence=False, mean=self.mean, std=self.std)
                #NOTE: scaleFactor, x_displacement, y_displacement not used if croppedVariations
                
                #Flatten
                #keypoints_original_norm_noconfidence_flat = [item for sublist in keypoints_original_norm for item in sublist]
                #keypoints_original_norm_noconfidence_flatFloat = [float(k) for k in keypoints_original_norm_noconfidence_flat]
                
                #keypoints_original_norm_noconfidence_flatFloat = poseUtils.keypointsListFlatten(keypoints_original_norm)

                keypoints_original_flat = torch.tensor(keypoints_original_norm)
                


                #keypoints_original_flat = torch.tensor(keypoints_original_norm_noconfidence_flat)
                
                #keypoints_original_norm, dummy, dummy, dummy = openPoseUtils.normalize(buffered_keypoints_original, BodyModelOPENPOSE15, keepConfidence=False)
                #keypoints_original_norm_noconfidence_flat = [item for sublist in keypoints_original_norm for item in sublist]
                #keypoints_original_flat = torch.tensor(keypoints_original_norm_noconfidence_flat)
              	
                if self.croppedVariations:
                	variations = openPoseUtils.crop(buffered_keypoints_original, BodyModelOPENPOSE15)               	
                else:
                	#no variations, only the original one
                	variations = [buffered_keypoints_original]

                for v_idx, keypoints_cropped in enumerate(variations):    
                    #The normalization is performed over the cropped skeleton

                    keypoints_croppedNoConf, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
                    keypoints_cropped_norm, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalizeV2(keypoints_croppedNoConf, BodyModelOPENPOSE15, self.normalizationStrategy, keepConfidence=False, mean=self.mean, std=self.std)              
                    
                    
                    #k = 10
                    #print("---------------------------------")
                    #print("RBefore: keypoints_cropped[0]", keypoints_cropped[k])
                    #Rkeypoints_cropped_norm, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalize(keypoints_cropped, BodyModelOPENPOSE15, keepConfidence=True)   
                    #print("RNormalized: keypoints_croppedNoConf[0]", Rkeypoints_cropped_norm[k])
                    #Rkeypoints_croppedNoConf, confidence_values = openPoseUtils.removeConfidence(Rkeypoints_cropped_norm)
                    #print("RRemoved conf: keypoints_croppedNoConf[0]", Rkeypoints_croppedNoConf[k])

                    #print("Before: keypoints_cropped[0]", keypoints_cropped[k])
                    #keypoints_croppedNoConf, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
                    #print("Removed conf: keypoints_croppedNoConf[0]", keypoints_croppedNoConf[k])
                    #keypoints_croppedNoConf, scaleFactor, x_displacement, y_displacement = openPoseUtils.normalize(keypoints_croppedNoConf, BodyModelOPENPOSE15, keepConfidence=False)   
                    #print("Normalized: keypoints_croppedNoConf[0]", keypoints_croppedNoConf[k])
                    #print("---------------------------------")

                    #keypoints_croppedFlat = [item for sublist in keypoints_cropped_norm for item in sublist]
                    #keypoints_croppedFlatFloat = [float(k) for k in keypoints_croppedFlat]
                    keypoints_croppedFlatFloatTensor = torch.tensor(keypoints_cropped_norm)
                    
                    #The confidence_values of the cropped skeletons signal which bones to fake and which to keep
                    confidence_values = torch.tensor(confidence_values)

                    buffer_variations.append((keypoints_croppedFlatFloatTensor, keypoints_original_flat, confidence_values, scaleFactor, x_displacement, y_displacement, "unknown file"))
            except:
                print("ERROR: Keypoints not normalized")


        len_variations = len(buffer_variations)
        print("Variations generated: ", len_variations)
        print("Shuffling variations...")
        random.shuffle(buffer_variations)
        print("Yielding variations...")
        for tup in buffer_variations:
            yield tup[0], tup[1], tup[2], tup[3], tup[4], tup[5], tup[6]
        print(str(len_variations)+" variations yield.")
        buffer_variations = []
        buffer_originals = []
            
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
