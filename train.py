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
import matplotlib.pyplot as plt #if raises warning edit .matplotlib/matplotlibrc and change backend for Agg
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
import dataset
import datasetBasic
import datasetH36M
import BodyModelOPENPOSE15
import BodyModelOPENPOSE25
import time
import openPoseUtils
import importlib
import train_utils
import colors
from datetime import date

def prepare_dataset(croppedVariations = True, normalizationStrategy = "center_scale", mean=None, std=None, max_len_buffer_originals = None, DISCARDINCOMPLETEPOSES=False): 
    jsonDataset = datasetModule.JsonDataset(inputpath_cropped=DATASET_CROPPED, inputpath_original=DATASET_ORIGINAL, bodyModel=BODY_MODEL, croppedVariations=croppedVariations, normalizationStrategy=normalizationStrategy, mean=mean, std=std, max_len_buffer_originals = max_len_buffer_originals, DISCARDINCOMPLETEPOSES=DISCARDINCOMPLETEPOSES)  
    dataloader, dataloaderValidation = train_utils.split_dataset(jsonDataset, TRAINSPLIT, batch_size, workers)
    return dataloader, dataloaderValidation


if __name__ == "__main__":
    VERSION=str(date.today())

    argv = sys.argv
    try:
        DATASET_CROPPED=argv[1]
        DATASET_ORIGINAL=argv[2]
        OUTPUTPATH=argv[3]
        DATASET_CHARADE=argv[4]
        DATASET_CHARADE_IMAGES=argv[5]
        DATASET_TEST=argv[6]
        DATASET_TEST_IMAGES=argv[7]
        BODY_MODEL = eval(argv[8])
        datasetModule = eval(argv[9])
        MODEL=argv[10]
        NORMALIZATION=argv[11]
        if argv[12]=="0":
            KEYPOINT_RESTORATION=False
        else:
            KEYPOINT_RESTORATION=True
        LEN_BUFFER_ORIGINALS=int(argv[13])
        if argv[14]=="0":
            CROPPED_VARIATIONS=False
        else:
            CROPPED_VARIATIONS=True
        NZ=int(argv[15])
        if argv[16]=="0":
            DISCARDINCOMPLETEPOSES=False
        else:
            DISCARDINCOMPLETEPOSES=True
        TRAINSPLIT=argv[17]
        PIXELLOSS_WEIGHT=int(argv[18])

    except ValueError:
        print("Wrong arguments. Expecting two paths.")

    models = importlib.import_module(MODEL)

    ####### INITIAL WARNINGS ########
    if not CROPPED_VARIATIONS:
        print(colors.CRED + "CROPPED_VARIATIONS=" + str(CROPPED_VARIATIONS) + colors.CEND)
    else:
        print(colors.CGREEN + "CROPPED_VARIATIONS=" + str(CROPPED_VARIATIONS) + colors.CEND)

    if not NORMALIZATION=="center_scale":
        print(colors.CRED + "NORMALIZATION=" + str(NORMALIZATION) + colors.CEND)
    else:
        print(colors.CGREEN + "NORMALIZATION=" + str(NORMALIZATION) + colors.CEND)
    if LEN_BUFFER_ORIGINALS<65536:
        print(colors.CRED + "LEN_BUFFER_ORIGINALS=" + str(LEN_BUFFER_ORIGINALS) + colors.CEND)
    else:
        print(colors.CGREEN + "LEN_BUFFER_ORIGINALS=" + str(LEN_BUFFER_ORIGINALS) + colors.CEND)
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
    if not DISCARDINCOMPLETEPOSES:
        print(colors.CRED + "DISCARDINCOMPLETEPOSES=" + str(DISCARDINCOMPLETEPOSES) + colors.CEND)
    else:
        print(colors.CGREEN + "DISCARDINCOMPLETEPOSES=" + str(DISCARDINCOMPLETEPOSES) + colors.CEND)
    ###########

    pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 
    pathlib.Path(OUTPUTPATH+"/model/").mkdir(parents=True, exist_ok=True) 

    run_info_file_path = "/run_info.json"

    #To avoid parallel error on macos set workers = 0
    #Number of workers for dataloader
    workers = 1 #4

    #os.environ['OMP_NUM_THREADS'] = "1" 
    #os.environ['OMP_NUM_THREADS'] = "1"
    #print("WARNING: Disabling paralelism to avoid error in macOS")
    #Also run export OMP_NUM_THREADS=1 in the terminal

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=int, default = 0)
    parser.add_argument('--interactive', type=int, default = 0)
    arguments, unparsed = parser.parse_known_args()

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    # Batch size during training
    #batch_size = 128
    #batch_size = 64
    batch_size = 128#128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    numJoints = len(BODY_MODEL.POSE_BODY_25_BODY_PARTS)  #15
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
    ngpu = 1



    '''
    dataloader = prepare_dataset(croppedVariations = False, normalizationStrategy = "none", mean=None, std=None, max_len_buffer_originals = LEN_BUFFER_ORIGINALS)
    print("Computing mean and std...")
    mean, std = train_utils.get_mean_and_std(dataloader)
    '''
    mean=  466.20676
    std=  114.26538

    print("mean= ", mean)
    print("std= ", std)

    #dataloader = prepare_dataset(croppedVariations = True, normalizationStrategy = "center_scale")
    dataloader, dataloaderValidation = prepare_dataset(croppedVariations = CROPPED_VARIATIONS, normalizationStrategy = NORMALIZATION, mean=mean, std=std, max_len_buffer_originals = LEN_BUFFER_ORIGINALS, DISCARDINCOMPLETEPOSES=DISCARDINCOMPLETEPOSES)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Device detected: ", device)

    theModels = models.Models(ngpu, numJoints, nz, KEYPOINT_RESTORATION, device)
    theTrainSetup = models.TrainSetup(theModels, ngpu, numJoints, nz, lr, beta1, PIXELLOSS_WEIGHT, device)


    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(batch_size, nz, device=device)

    # Lists to keep track of progress
    img_list = []
    #G_losses = []
    #D_losses = []
    iters = 0

    #Setup tensorboard
    tb = SummaryWriter()
    log_dir = tb.get_logdir()

    #train_utils.writeRunInfoFile(log_dir, DATASET_CROPPED, DATASET_ORIGINAL, NORMALIZATION, VERSION, str(batch_size))
    run_info_json = {}
    run_info_json["id"] = log_dir
    run_info_json["DATASET_CROPPED"] = DATASET_CROPPED
    run_info_json["DATASET_ORIGINAL"] = DATASET_ORIGINAL
    run_info_json["normalization"] = NORMALIZATION
    run_info_json["model version"] = VERSION
    run_info_json["batch size"] = str(batch_size)
    run_info_json["lr"] = str(lr)
    run_info_json["results"] = []
    train_utils.writeRunInfoFile(run_info_json, run_info_file_path, OUTPUTPATH)

    ####################################################
    # Training Loop
    ####################################################
    print("Starting Training Loop...")
    epoch_idx = 0
    MAX_BATCHES = 25000
    # For each epoch (all batches -> all data)
    step_absolute = 0
    step_absolute_validation = 0
    for epoch in range(num_epochs):        
        print("EPOCH ", epoch)
        # For each batch in the dataloader
        print("Iterating batches...")
        i = 0
        for batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file in dataloader:
            start = time.time()
            if epoch_idx == 0 and i == 0:
                print("INFORMATION FOR THE FIRST BATCH")
                print("batch_of_keypoints_cropped.shape", batch_of_keypoints_cropped.shape)
                print("batch_of_keypoints_original.shape", batch_of_keypoints_original.shape)
                print("confidence_values.shape", confidence_values.shape)
                print("Batch size received: ", batch_of_keypoints_cropped.size(0))

            if i > MAX_BATCHES:
                break

            batch_of_keypoints_cropped = batch_of_keypoints_cropped.to(device)
            batch_of_keypoints_original = batch_of_keypoints_original.to(device)
            b_size = batch_of_keypoints_cropped.size(0)

            #train step
            models.trainStep(theModels, theTrainSetup, b_size, device, tb, step_absolute, num_epochs, epoch, i, batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file)

            # Save model and display the first 64 results each 1000 batches
            if i % 1000 == 0: 
                #save model
                print("1000 steps, saving model...")
                models.save(theModels, OUTPUTPATH, epoch, i)

                #run_info_json["results"].append({'epoch': epoch, 'batch': i, 'lossG':errG.item()})
                #train_utils.writeRunInfoFile(run_info_json, run_info_file_path, OUTPUTPATH)

                #display the batch
                #originalReshapedAsKeypoints = np.reshape(batch_of_keypoints_original.cpu(), (batch_size, numJoints, 2))
                #croppedReshapedAsKeypoints = np.reshape(batch_of_keypoints_cropped.cpu(), (batch_size, numJoints, 2))
                #croppedReshapedAsKeypoints = croppedReshapedAsKeypoints.numpy()
                originalReshapedAsKeypoints = batch_of_keypoints_original
                croppedReshapedAsKeypoints = batch_of_keypoints_cropped

                fakeReshapedAsKeypoints = models.inference(theModels, b_size, fixed_noise, numJoints, batch_of_keypoints_cropped, confidence_values)

                #print("fakeReshapedAsKeypoints:")
                #print(fakeReshapedAsKeypoints[0])

                train_utils.drawGrid(fakeReshapedAsKeypoints, originalReshapedAsKeypoints, croppedReshapedAsKeypoints, OUTPUTPATH, batch_size, BODY_MODEL, scaleFactor.numpy(), x_displacement.numpy(), y_displacement.numpy(), batch_of_json_file, NORMALIZATION, mean, std)
        
            iters += 1
            
            i += 1
            step_absolute += 1
            end = time.time()
            print(f"Training step took {end - start}")
                       
        print("---- end of epoch "+str(epoch)+"---")
        '''
        if batch_of_keypoints_cropped.size(0) < batch_size:
            print("FATAL ERROR: Batch size = ", batch_of_keypoints_cropped.size(0))
            sys.exit()
        '''
        epoch_idx += 1

        #VALIDATION
        '''
        for batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file in dataloaderValidation:
            with torch.no_grad():
                # Generate fake image batch with G
                batch_of_fake_original = netG(batch_of_keypoints_cropped, noise)

                #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
                if KEYPOINT_RESTORATION:
                    batch_of_fake_original = models.restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, confidence_values)

                g_adv_validation = lossFunctionG_adversarial(output, label) #adversarial loss
                g_pixel_validation = lossFunctionG_regression(batch_of_fake_original, batch_of_keypoints_original) #pixel loss
                errG_validation = 0 * g_adv_validation + 1 * g_pixel_validation
        tb.add_scalar("LossG_validation", errG_validation.item(), step_absolute_validation)
        print("---->VALIDATION LOSS = ", errG_validation.item())
        step_absolute_validation = step_absolute_validation+1
        #ENDVALIDATION
        '''
    tb.close()
    print("Finshed. epochs = ", epoch_idx)
