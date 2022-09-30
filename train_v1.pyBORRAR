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
batch_size = 128#32#128#128#128#64

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


def prepare_dataset(croppedVariations = True, normalizationStrategy = "center_scale", mean=None, std=None, max_len_buffer_originals = None, DISCARDINCOMPLETEPOSES=False): 
    jsonDataset = datasetModule.JsonDataset(inputpath_cropped=DATASET_CROPPED, inputpath_original=DATASET_ORIGINAL, bodyModel=BODY_MODEL, croppedVariations=croppedVariations, normalizationStrategy=normalizationStrategy, mean=mean, std=std, max_len_buffer_originals = max_len_buffer_originals, DISCARDINCOMPLETEPOSES=DISCARDINCOMPLETEPOSES)  
    dataloader, dataloaderValidation = train_utils.split_dataset(jsonDataset, TRAINSPLIT, batch_size, workers)
    return dataloader, dataloaderValidation

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


#pytorchUtils.explainDataloader(dataloader)
##########################


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Device detected: ", device)

# Create the generator
netG_ = models.Generator(ngpu, numJoints, nz)
netG = netG_.to(device)

#Register my debug hook
if arguments.debug:
	pytorchUtils.registerDebugHook(netG_)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(train_utils.weights_init)

# Print the model
print(netG)
#pytorchUtils.explainModel(netG, 1, 1, 28, 28)
#pytorchUtils.computeModel(netG, 1, [{"layer":0, "output":7},{"layer":6, "output":14},{"layer":9, "output":28}])

# Create the Discriminator
netD_ = models.Discriminator(ngpu, numJoints)
netD = netD_.to(device)

#Register my debug hook
if arguments.debug:
	pytorchUtils.registerDebugHook(netD_)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(train_utils.weights_init)

# Print the model
print(netD)
pytorchUtils.explainModel(netD, 28, 28, 1, 1)

# Initialize BCELoss function
#criterion = nn.BCELoss() according to https://github.com/soumith/ganhacks/issues/363
#criterion = torch.nn.BCEWithLogitsLoss
#criterion = nn.BCELoss() 
lossFunctionD = nn.BCELoss() 
lossFunctionG_adversarial = nn.BCELoss() 
lossFunctionG_regression = torch.nn.MSELoss()#torch.nn.MSELoss() #torch.nn.L1Loss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batch_size, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))




# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
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

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        ## Train with all-real batch
        netD.zero_grad()
     
     	#Batch of real labels
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        # Forward pass real batch through D
        output = netD(batch_of_keypoints_cropped, batch_of_keypoints_original).view(-1)
        
        # Calculate loss on all-real batch
        errD_real = lossFunctionD(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        
        # Generate fake image batch with G
        batch_of_fake_original = netG(batch_of_keypoints_cropped, noise)

        #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
        if KEYPOINT_RESTORATION:
            batch_of_fake_original = models.restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, confidence_values)

        #As they are fake images let's prepare a batch of labels FAKE
        label.fill_(fake_label)
       
        # Classify all fake batch with D
        output = netD(batch_of_keypoints_cropped, batch_of_fake_original.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = lossFunctionD(output, label)
        
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        
        D_G_z1 = output.mean().item()
        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        
        label.fill_(real_label)  # fake labels are real for generator cost
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(batch_of_keypoints_cropped, batch_of_fake_original).view(-1)
        
        # Calculate G's loss based on this output
        #errG = criterion(output, label)

        ##############

        g_adv = lossFunctionG_adversarial(output, label) #adversarial loss
        g_pixel = lossFunctionG_regression(batch_of_fake_original, batch_of_keypoints_original) #pixel loss

        errG = (1-PIXELLOSS_WEIGHT) * g_adv + PIXELLOSS_WEIGHT * g_pixel

        
        ###############

        # Calculate gradients for G
        errG.backward()
        
        D_G_z2 = output.mean().item()
        
        # Update G
        optimizerG.step()

        tb.add_scalar("LossG", errG.item(), step_absolute)
        tb.add_scalar("g_adv", g_adv.item(), step_absolute)
        tb.add_scalar("g_pixel", g_pixel.item(), step_absolute)
        tb.add_scalar("LossD", errD.item(), step_absolute)
        tb.add_scalar("errD_real", errD_real.item(), step_absolute)
        tb.add_scalar("errD_fake", errD_fake.item(), step_absolute)

        # Output training stats each 50 batches
        if i % 50 == 0:
            print("**************************************************************")
            print('[%d/%d][%d/?]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, #len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('errD_real: %.4f, errD_fake: %.4f\t'
                  % (errD_real.item(), errD_fake.item()))

            print('loss g_adv: %.4f, loss g_pixel: %.4f\t'
                  % (g_adv.item(), g_pixel.item()))

        # Save model and display the first 64 results each 1000 batches
        if i % 1000 == 0: 
            print("1000 steps, saving model...")
            torch.save(netG.state_dict(), OUTPUTPATH+"/model/model_epoch"+str(epoch)+"_batch"+str(i)+".pt")

            run_info_json["results"].append({'epoch': epoch, 'batch': i, 'lossG':errG.item()})
            train_utils.writeRunInfoFile(run_info_json, run_info_file_path, OUTPUTPATH)

            with torch.no_grad():
                print("drawing batch...")
                fake = netG(batch_of_keypoints_cropped, fixed_noise).detach().cpu()
                #We restore the original keypoints (before denormalizing)
                if KEYPOINT_RESTORATION:
                    fake = models.restoreOriginalKeypoints(fake, batch_of_keypoints_cropped, confidence_values)
                print("Shape of fake: ", fake.shape)
                fakeReshapedAsKeypoints = np.reshape(fake, (batch_size, numJoints, 2))
                fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
                originalReshapedAsKeypoints = np.reshape(batch_of_keypoints_original.cpu(), (batch_size, numJoints, 2))
                croppedReshapedAsKeypoints = np.reshape(batch_of_keypoints_cropped.cpu(), (batch_size, numJoints, 2))
                croppedReshapedAsKeypoints = croppedReshapedAsKeypoints.numpy()

                train_utils.drawGrid(fakeReshapedAsKeypoints, originalReshapedAsKeypoints, croppedReshapedAsKeypoints, OUTPUTPATH, batch_size, BODY_MODEL, scaleFactor, x_displacement, y_displacement, batch_of_json_file, NORMALIZATION, mean, std)

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

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
