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
import datasetH36M_noCroppingVariations as datasetH36M
import time
import Configuration
import openPoseUtils
import importlib

from torch.nn import functional as F


VERSION="13"
NORMALIZATION="SCALE"

argv = sys.argv
try:
    DATASET_CROPPED=argv[1]
    DATASET_ORIGINAL=argv[2]
    OUTPUTPATH=argv[3]
    DATASET_CHARADE=argv[4]
    DATASET_CHARADE_IMAGES=argv[5]
    DATASET_TEST=argv[6]
    DATASET_TEST_IMAGES=argv[7]
    conf = Configuration.Configuration()
    conf.set_BODY_MODEL(argv[8])
    datasetModule = eval(argv[9])
    MODEL=argv[10]
    #datasetModule = eval("datasetH36M")
    #print(conf.bodyModel.POSE_BODY_25_BODY_PARTS_DICT[20])
except ValueError:
    print("Wrong arguments. Expecting two paths.")

models = importlib.import_module(MODEL)

# Root directory for dataset
#dataroot_cropped = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format_cropped"
#dataroot_original = "/Volumes/ElementsDat/pose/COCO/ruben_structure/keypoints_openpose_format"


#For visualizing result
#(ORIGINAL_IMAGES_PATH = "data/H36M_ECCV18/Train/IMG"
#ORIGINAL_IMAGES_PATH = "/Volumes/ElementsDat/pose/COCO/train2017"

#CROPPED_IMAGES_PATH = "data/H36M_ECCV18/Train/IMG_CROPPED"
#CROPPED_IMAGES_PATH = "/Volumes/ElementsDat/pose/H36M/ECCV2018/ECCV18_Challenge/Train/IMG"

#OPENPOSE_IMAGES_KEYPOINTS = "data/H36M_ECCV18/Train/result"
#OPENPOSE_IMAGES_KEYPOINTS = "/Volumes/ElementsDat/pose/H36M/ECCV2018/keyponts_generated_by_openpose_for_train_images_cropped"

#OUTPUTPATH = "data/output2"
pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 
pathlib.Path(OUTPUTPATH+"/model/").mkdir(parents=True, exist_ok=True) 
#pathlib.Path(OUTPUTPATH+"/Test/").mkdir(parents=True, exist_ok=True) 
#pathlib.Path(OUTPUTPATH+"/Test/keypoints").mkdir(parents=True, exist_ok=True) 
#pathlib.Path(OUTPUTPATH+"/Test/images").mkdir(parents=True, exist_ok=True) 

run_info_file_path = "/run_info.json"

def writeRunInfoFile(run_info_json):
    run_info_file = open(OUTPUTPATH+run_info_file_path, 'w')
    json.dump(run_info_json, run_info_file)
    run_info_file.flush()
    run_info_file.close()

# Validating with the Charada dataset
#dataroot_validation = "/Users/rtous/DockerVolume/charade/input/keypoints"
#TEST_IMAGES_PATH = "/Users/rtous/DockerVolume/charade/input/images"



#To avoid parallel error on macos set workers = 0
# Number of workers for dataloader
#workers = 0 #2
workers = 1 #4

#os.environ['OMP_NUM_THREADS'] = "1" 
#print("WARNING: Disabling paralelism to avoid error in macOS")
#Also run export OMP_NUM_THREADS=1 in the terminal

#Not used
'''
def countFiles(dirpath, endswith):
    count = 0
    scandirIterator = os.scandir(dirpath)
    for item in scandirIterator:
        if item.name.endswith(endswith):
            count += 1
    return count
    scandirIterator.close()
'''



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
batch_size = 64#128#128#64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.

numJoints = len(conf.bodyModel.POSE_BODY_25_BODY_PARTS)  #15
image_size = numJoints*2

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 10

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

def drawBatch(fakeReshapedAsKeypoints, originalReshapedAsKeypoints):
    NUM_ROWS = 8
    NUM_COLS = 8
    WIDTH = 64
    HEIGHT = 64
    imagesCropped = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
    images = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
    imagesOriginal = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
    
    ####### DRAW DEBUG POSES FOR THE FIRST 64 IMAGES
    print("Draw debug poses for the batch")
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
        

        #??????????????????????
        '''
        fakeKeypointsOneImage, dummy, dummy, dummy = openPoseUtils.normalize(fakeKeypointsOneImage)
        if (idx == 0):
            print("normalizedFakeKeypointsOneImage (output normalized):", fakeKeypointsOneImage)
        '''
        
        #fakeKeypointsCroppedOneImageInt = poseUtils.keypointsToInteger(fakeKeypointsCroppedOneImage)
        #fakeKeypointsOneImageInt = poseUtils.keypointsToInteger(fakeKeypointsOneImage)
        originalReshapedAsKeypointsOneImageInt = originalReshapedAsKeypointsOneImage
        fakeKeypointsCroppedOneImageInt = fakeKeypointsCroppedOneImage
        fakeKeypointsOneImageInt = fakeKeypointsOneImage
        
        
        #Draw result over the original image
        fakeKeypointsCroppedOneImageIntRescaled = openPoseUtils.denormalize(fakeKeypointsOneImageInt, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage)
        
        #fakeKeypointsCroppedOneImageIntRescaledNP = poseUtils.keypoints2Numpy(fakeKeypointsCroppedOneImageIntRescaled)
        #fakeKeypointsCroppedOneImageIntRescaledNP = poseUtils.scale(fakeKeypointsCroppedOneImageIntRescaledNP, 0.01)


        #If we want to save the .json files of the batch
        #openPoseUtils.keypoints2json(fakeKeypointsOneImageInt, OUTPUTPATH+"/"+f"{idx:02d}"+"_img_keypoints.json")
        
        json_file_without_extension = os.path.splitext(json_file)[0]
        json_file_without_extension = json_file_without_extension.replace('_keypoints', '')
        
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
            poseUtils.draw_pose_scaled_centered(blank_imageOriginal, originalReshapedAsKeypointsOneImageInt.numpy(), -1, conf.bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 1/(WIDTH/4), WIDTH/2, HEIGHT/2, 8)
            poseUtils.draw_pose_scaled_centered(blank_imageCropped, fakeKeypointsCroppedOneImageInt, -1, conf.bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 1/(WIDTH/4), WIDTH/2, HEIGHT/2, 8)
            poseUtils.draw_pose_scaled_centered(blank_image, fakeKeypointsOneImageInt, -1, conf.bodyModel.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 1/(WIDTH/4), WIDTH/2, HEIGHT/2, 8)
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

def prepare_dataset():
  
    #jsonDataset = dataset.JsonDataset(inputpath_cropped=DATASET_CROPPED, inputpath_original=DATASET_ORIGINAL)

    jsonDataset = datasetModule.JsonDataset(inputpath_cropped=DATASET_CROPPED, inputpath_original=DATASET_ORIGINAL, bodyModel=conf.bodyModel)

    dataloader = torch.utils.data.DataLoader(jsonDataset, batch_size=batch_size, 
                                             num_workers=workers)

    # Batch and shuffle data with DataLoader
    #trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    return dataloader

dataloader = prepare_dataset()

#pytorchUtils.explainDataloader(dataloader)
##########################


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print("Device detected: ", device)
'''
# Plot some training images
print("Plotting some...")
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
'''

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, numJoints*2), reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    #torch.nn.MSELoss()
    BCE = F.mse_loss(recon_x, x, size_average=None, reduce=None, reduction='sum')#mean

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
    #return 10000*(BCE + KLD)
    #Here suggests to change this: https://www.quora.com/How-do-you-fix-a-Variational-Autoencoder-VAE-that-suffers-from-mode-collapse



model = models.VAE(numJoints*2, nz).to(device)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batch_size, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

#Ruben
#plt.ion() # turn on interactive mode, non-blocking `show`
#plt.ion()
#plt.show()

#Aproach pyformulas
'''
fig = plt.figure()
canvas = np.zeros((480,640))
screen = pf.screen(canvas, 'Sinusoid')
start = time.time()
'''

    
tb = SummaryWriter()
log_dir = tb.get_logdir()
run_info_json = {}
run_info_json["id"] = log_dir
run_info_json["DATASET_CROPPED"] = DATASET_CROPPED
run_info_json["DATASET_ORIGINAL"] = DATASET_ORIGINAL
run_info_json["normalization"] = NORMALIZATION
run_info_json["model version"] = VERSION
run_info_json["batch size"] = str(batch_size)
run_info_json["lr"] = str(lr)
run_info_json["results"] = []
writeRunInfoFile(run_info_json)

print("Starting Training Loop...")
epoch_idx = 0
MAX_BATCHES = 25000
# For each epoch
step_absolute = 0
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

        #print("received: ", batch_of_keypoints_cropped[0])

        batch_of_keypoints_cropped = batch_of_keypoints_cropped.to(device)
        batch_of_keypoints_original = batch_of_keypoints_original.to(device)
        b_size = batch_of_keypoints_cropped.size(0)
    
        #CVAE code
        recon_batch, mu, logvar = model(batch_of_keypoints_original)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, batch_of_keypoints_original, mu, logvar)
        loss.backward()
        #train_loss += loss.detach().cpu().numpy()
        optimizer.step()

        tb.add_scalar("Loss", loss.item(), step_absolute)

        # Output training stats each 50 batches
        if i % 50 == 0:
            print("**************************************************************")
            print('[%d/%d][%d/?]\tLoss: %.4f' % (epoch, num_epochs, i, loss.item()))
        
        if i % 1000 == 0: 
            torch.save(model.state_dict(), OUTPUTPATH+"/model/model_epoch"+str(epoch)+"_batch"+str(i)+".pt")
            with torch.no_grad():
                #c = torch.eye(10, 10)
                #sample = torch.randn(10, 20).to(device)
                fake = model.decode(fixed_noise).cpu()
                #save_image(sample.view(10, 1, 28, 28),
                #           'data/output/VAE/sample_' + str(epoch) + '.png')
                
                fakeReshapedAsKeypoints = np.reshape(fake, (batch_size, numJoints, 2))
                fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
                originalReshapedAsKeypoints = np.reshape(batch_of_keypoints_original.cpu(), (batch_size, numJoints, 2))
                croppedReshapedAsKeypoints = np.reshape(batch_of_keypoints_cropped.cpu(), (batch_size, numJoints, 2))
                croppedReshapedAsKeypoints = croppedReshapedAsKeypoints.numpy()

                drawBatch(fakeReshapedAsKeypoints, originalReshapedAsKeypoints)

        iters += 1
        
        i += 1
        step_absolute += 1
        end = time.time()
        print(f"Training step took {end - start}")
                   
    print("---- end of epoch "+str(epoch)+"---")
    epoch_idx += 1
tb.close()
print("Finshed. epochs = ", epoch_idx)

