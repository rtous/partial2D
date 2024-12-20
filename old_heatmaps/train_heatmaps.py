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
import models_heatmaps
import dataset
import datasetH36M_heatmaps
import normalization_heatmaps as normalization
import time
import BodyModelOPENPOSE15

CRED = '\033[91m'
CEND = '\033[0m'

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
    HEATMAP_SIZE=int(argv[8])

except ValueError:
    print("Wrong arguments. Expecting two paths.")

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
batch_size = 64#128#64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 30

# Number of channels in the training images. For color images this is 3
nc = 15

# Size of z latent vector (i.e. size of generator input)
nz = 100

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

def prepare_dataset():
  
    #jsonDataset = dataset.JsonDataset(inputpath_cropped=DATASET_CROPPED, inputpath_original=DATASET_ORIGINAL)

    jsonDataset = datasetH36M_heatmaps.JsonDataset(inputpath_cropped=DATASET_CROPPED, inputpath_original=DATASET_ORIGINAL, HEATMAP_SIZE=HEATMAP_SIZE)

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


# Create the generator
netG_ = models_heatmaps.Generator(channels=nc)
netG = netG_.to(device)

#Register my debug hook
if arguments.debug:
	pytorchUtils.registerDebugHook(netG_)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)
#pytorchUtils.explainModel(netG, 1, 1, 28, 28)
#pytorchUtils.computeModel(netG, 1, [{"layer":0, "output":7},{"layer":6, "output":14},{"layer":9, "output":28}])

# Create the Discriminator
netD_ = models_heatmaps.Discriminator64(ngpu)
netD = netD_.to(device)

#Register my debug hook
if arguments.debug:
	pytorchUtils.registerDebugHook(netD_)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)
pytorchUtils.explainModel(netD, 28, 28, 1, 1)

# Initialize BCELoss function
#criterion = nn.BCELoss() according to https://github.com/soumith/ganhacks/issues/363
#criterion = torch.nn.BCEWithLogitsLoss
#criterion = nn.BCELoss() 
lossFunctionD = nn.BCELoss() 
lossFunctionG_adversarial = nn.BCELoss()#nn.MSELoss() 
lossFunctionG_regression = torch.nn.MSELoss()#L1Loss()
#lossFunctionG_regression = torch.nn.BCELoss(reduction='none')
#lossFunctionG_regression = torch.nn.BCELoss()
#Regression loss whould be: BCELoss(reduction='none')

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batch_size, nz, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


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

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
     
     	#Batch of real labels
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        
        #print("***********************")
        #print("***********************")
        #print("batch_of_keypoints_cropped.shape=",batch_of_keypoints_cropped.shape)
        #print("batch_of_keypoints_original.shape=",batch_of_keypoints_original.shape)
        #print("***********************")
        #print("***********************")

        #print("batch_of_keypoints_cropped.is_cuda=",batch_of_keypoints_cropped.is_cuda)
        #print("batch_of_keypoints_original.is_cuda=",batch_of_keypoints_original.is_cuda)

        #print("netD=",next(netD.parameters()).is_cuda)

        # Forward pass real batch through D
        # In a conditional GAN the input of the Discriminator not just the ouput, is a valid pair (condition-output)
        print(CRED + "WARNING: Discriminator disabled (DEBUGGING)" + CEND)
        print(CRED + "WARNING: Generator loss changed, and sigmoid instad of tanh too (DEBUGGING)" + CEND)
        
        '''
        startD = time.time()
        output = netD(batch_of_keypoints_cropped, batch_of_keypoints_original).view(-1)
        endD = time.time()
        print(f"Discriminator took {endD - startD}")
        #output = netD(batch_of_keypoints_cropped, batch_of_keypoints_original)
        
        #print("Discriminator output: ", output.shape)
        
        # Calculate loss on all-real batch
        errD_real = lossFunctionD(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()
        '''

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        #print("Noise shape: ", noise.shape)
        
        # Generate fake image batch with G
        #batch_of_fake_original = netG(batch_of_keypoints_cropped, noise)
        startG = time.time()
        #batch_of_fake_original = netG(batch_of_keypoints_cropped, noise)
        batch_of_fake_original = netG(batch_of_keypoints_cropped)
        endG = time.time()
        print(f"Generator took {endG - startG}")
        #print("Generator output shape: ", batch_of_fake_original.shape)

        #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
        #This is done over normalized keypoints
        
        print(CRED + "WARNING: keypoints restauration disabled (DEBUGGING)" + CEND)
        print(CRED + "(when testing too)" + CEND)
        
        batch_of_fake_original = models_heatmaps.restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, confidence_values)

        #As they are fake images let's prepare a batch of labels FAKE
        label.fill_(fake_label)
        '''
        # Classify all fake batch with D
        # In a conditional GAN the input of the Discriminator not just the ouput, is a valid pair (condition-output)
        output = netD(batch_of_keypoints_cropped, batch_of_fake_original.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = lossFunctionD(output, label)
        
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        startDu = time.time()
        errD_fake.backward()
        
        D_G_z1 = output.mean().item()
        
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        
        # Update D
        
        optimizerD.step()
        endDu = time.time()
        print(f"optimizerD.step() took {endDu - startDu}")
        '''
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        
        label.fill_(real_label)  # fake labels are real for generator cost
        
        # Since we just updated D, perform another forward pass of all-fake batch through D
        #output = netD(batch_of_keypoints_cropped, batch_of_fake_original).view(-1)
        
        # Calculate G's loss based on this output
        #errG = criterion(output, label)

        ##############

        #g_adv = lossFunctionG_adversarial(output, label) #adversarial loss
        g_pixel = lossFunctionG_regression(batch_of_fake_original, batch_of_keypoints_original) #pixel loss

        #errG = 0.25 * g_adv + 0.75 * g_pixel

        #errG = 0 * g_adv + 1 * g_pixel
        errG = g_pixel

        #errG = 0.001 * g_adv + 0.999 * g_pixel
        
        ###############
        startGu = time.time()
        # Calculate gradients for G
        errG.backward()
        
        #D_G_z2 = output.mean().item()
        
        # Update G
        optimizerG.step()
        endGu = time.time()
        print(f"G update took {endGu - startGu}")


        tb.add_scalar("LossG", errG.item(), step_absolute)
        #tb.add_scalar("g_adv", g_adv.item(), step_absolute)
        tb.add_scalar("g_pixel", g_pixel.item(), step_absolute)
        #tb.add_scalar("LossD", errD.item(), step_absolute)
        #tb.add_scalar("errD_real", errD_real.item(), step_absolute)
        #tb.add_scalar("errD_fake", errD_fake.item(), step_absolute)

        # Output training stats each 50 batches
        if i % 50 == 0:
            '''
            #print("WARNING: generator noise disabled")
            print("**************************************************************")
            print('[%d/%d][%d/?]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, #len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('errD_real: %.4f, errD_fake: %.4f\t'
                  % (errD_real.item(), errD_fake.item()))

            print('loss g_adv: %.4f, loss g_pixel: %.4f\t'
                  % (g_adv.item(), g_pixel.item()))
            '''
            #print("WARNING: generator noise disabled")
            print("**************************************************************")
            print('[%d/%d][%d/?]\tLoss_G: %.4f\t'
                  % (epoch, num_epochs, i, #len(dataloader),
                     errG.item()))
            print('loss g_pixel: %.4f\t'
                  % (g_pixel.item()))
        # Save model and display the first 64 results each 1000 batches
        if i % 100 == 0: 
            torch.save(netG.state_dict(), OUTPUTPATH+"/model/model_epoch"+str(epoch)+"_batch"+str(i)+".pt")

            run_info_json["results"].append({'epoch': epoch, 'batch': i, 'lossG':errG.item()})
            writeRunInfoFile(run_info_json)

            with torch.no_grad():
                #fake = netG(batch_of_keypoints_cropped, fixed_noise).detach().cpu()
                fake = netG(batch_of_keypoints_cropped).detach().cpu()
                #We restore the original keypoints (before denormalizing)
                
                #fake = torch.tensor(normalization.nullPoseBatch(len(batch_of_keypoints_cropped), normalization.HEATMAP_WIDTH, batch_size))
                #onesConfidentValuesBatch = normalization.onesConfidentValuesBatch(len(batch_of_keypoints_cropped[0]), batch_size)
                
           
                fake = models_heatmaps.restoreOriginalKeypoints(fake, batch_of_keypoints_cropped, confidence_values)
                
                #print(fake[0][0])
                #print(batch_of_keypoints_cropped[0][0])
                #print(confidence_values[0])
                #fake = models_heatmaps.restoreOriginalKeypoints(batch_of_keypoints_cropped, batch_of_keypoints_cropped, confidence_values)
                
                #print("Shape of fake: ", fake.shape)
                #fakeReshapedAsKeypoints = np.reshape(fake, (batch_size, 25, 2))
                #fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
                #originalReshapedAsKeypoints = np.reshape(batch_of_keypoints_original.cpu(), (batch_size, 25, 2))
                #croppedReshapedAsKeypoints = np.reshape(batch_of_keypoints_cropped.cpu(), (batch_size, 25, 2))
                #croppedReshapedAsKeypoints = croppedReshapedAsKeypoints.numpy()
                

                #originalReshapedAsKeypoints = normalization.denormalizeBatch(batch_of_keypoints_original.cpu().numpy(), scaleFactor, x_displacement, y_displacement)
                #croppedReshapedAsKeypoints = normalization.denormalizeBatch(batch_of_keypoints_cropped.cpu().numpy(), scaleFactor, x_displacement, y_displacement)
                #fakeReshapedAsKeypoints = normalization.denormalizeBatch(fake.cpu().numpy(), scaleFactor, x_displacement, y_displacement)
                originalReshapedAsKeypoints = batch_of_keypoints_original
                croppedReshapedAsKeypoints = batch_of_keypoints_cropped
                fakeReshapedAsKeypoints = fake


            NUM_ROWS = 8
            NUM_COLS = 8
            WIDTH = 128
            HEIGHT = 128
            imagesCropped = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
            images = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
            imagesOriginal = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
            
            
            ####### DRAW DEBUG POSES FOR THE FIRST 64 IMAGES
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
                

                #originalReshapedAsKeypointsOneImageInt = originalReshapedAsKeypointsOneImage
                #fakeKeypointsCroppedOneImageInt = fakeKeypointsCroppedOneImage
                #fakeKeypointsOneImageInt = fakeKeypointsOneImage

                print("originalReshapedAsKeypointsOneImage.shape: ", originalReshapedAsKeypointsOneImage.shape)
                print(originalReshapedAsKeypointsOneImage)
                #originalReshapedAsKeypointsOneImageInt = openPoseUtils.denormalizeV2(originalReshapedAsKeypointsOneImage, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage, "heatmaps", keepConfidence=False, mean=0, std=0)#conf.norm)
                originalReshapedAsKeypointsOneImageInt = normalization.denormalize(originalReshapedAsKeypointsOneImage.numpy(), scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage)
                

                print("fakeKeypointsCroppedOneImage.shape: ", fakeKeypointsCroppedOneImage.shape)
                #fakeKeypointsCroppedOneImageInt = openPoseUtils.denormalizeV2(fakeKeypointsCroppedOneImage, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage, "heatmaps", keepConfidence=False, mean=0, std=0)#conf.norm)
                fakeKeypointsCroppedOneImageInt = normalization.denormalize(fakeKeypointsCroppedOneImage.numpy(), scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage)
                
                #fakeKeypointsOneImageInt = openPoseUtils.denormalizeV2(fakeKeypointsOneImage, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage, "heatmaps", keepConfidence=False, mean=0, std=0)#conf.norm)
                fakeKeypointsOneImageInt = normalization.denormalize(fakeKeypointsOneImage.numpy(), scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage)
                

               	#Draw result over the original image
                fakeKeypointsCroppedOneImageIntRescaled = fakeKeypointsOneImageInt
                #fakeKeypointsCroppedOneImageIntRescaled = openPoseUtils.denormalize(fakeKeypointsOneImageInt, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage)
               	
               	#If we want to save the .json files of the batch
                #openPoseUtils.keypoints2json(fakeKeypointsOneImageInt, OUTPUTPATH+"/"+f"{idx:02d}"+"_img_keypoints.json")
                
                json_file_without_extension = os.path.splitext(json_file)[0]
               	json_file_without_extension = json_file_without_extension.replace('_keypoints', '')
               	
               	#Draw the pairs  
                try:
                    #def draw_pose_scaled_centered(img, keypoints, threshold, keypoint_index_pairs, colors, haveThreshold, scaleFactor, centerX, centerY, centerKeypointIndex, thickness=1):            

                    poseUtils.draw_pose_scaled_centered(blank_imageOriginal, originalReshapedAsKeypointsOneImageInt, -1, BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 8, WIDTH/2, HEIGHT/2, 8)
                    poseUtils.draw_pose_scaled_centered(blank_imageCropped, fakeKeypointsCroppedOneImageInt, -1, BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 8, WIDTH/2, HEIGHT/2, 8)
                    poseUtils.draw_pose_scaled_centered(blank_image, fakeKeypointsOneImageInt, -1, BodyModelOPENPOSE15.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False, 8, WIDTH/2, HEIGHT/2, 8)
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
                targetFilePathOriginal = OUTPUTPATH+"/debug_original.jpg"
                targetFilePathCropped = OUTPUTPATH+"/debug_input.jpg"
                targetFilePath = OUTPUTPATH+"/debug.jpg"
                cv2.imwrite(targetFilePathCropped, total_imageCropped)
                cv2.imwrite(targetFilePath, total_image)
                cv2.imwrite(targetFilePathOriginal, total_imageOriginal)
            except Exception:
                print("WARNING: Cannot draw tile ")
                traceback.print_exc()

            #inference.testImage(netG, outputPath, "dynamicData/012.jpg", "dynamicData/012_keypoints.json")
            #inference.testMany(netG, DATASET_CHARADE, DATASET_CHARADE_IMAGES, OUTPUTPATH, "/CHARADE", ".png")
            #inference.testMany(netG, DATASET_TEST, DATASET_TEST_IMAGES, OUTPUTPATH, "/TEST", ".jpg")
            
        # Save Losses for plotting later
        G_losses.append(errG.item())
        #D_losses.append(errD.item())

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
tb.close()
print("Finshed. epochs = ", epoch_idx)
