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
import BodyModelOPENPOSE25
import modelsECCV18OP
import datasetBasic

CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS = 0.1

argv = sys.argv
try:
    dataroot_cropped=argv[1]
    dataroot_original=argv[2]
    OUTPUTPATH=argv[3]
    dataroot_validation=argv[4]
    TEST_IMAGES_PATH=argv[5]

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
#pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 
#pathlib.Path(OUTPUTPATH+"/Test/").mkdir(parents=True, exist_ok=True) 
pathlib.Path(OUTPUTPATH+"/Test/keypoints").mkdir(parents=True, exist_ok=True) 
pathlib.Path(OUTPUTPATH+"/Test/images").mkdir(parents=True, exist_ok=True) 

# Validating with the Charada dataset
#dataroot_validation = "/Users/rtous/DockerVolume/charade/input/keypoints"
#TEST_IMAGES_PATH = "/Users/rtous/DockerVolume/charade/input/images"



#To avoid parallel error on macos set workers = 0
# Number of workers for dataloader
#workers = 0 #2
workers = 4
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
        self.scandirIterator = os.scandir(self.inputpath_cropped)
        for item in self.scandirIterator:
            json_file = str(item.name)
            if json_file.endswith(".json"):
                try:
                    #print("Processing file: "+json_file)
                    keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(join(self.inputpath_cropped, json_file), BodyModelOPENPOSE25)
                    keypoints_cropped, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
                    keypoints_cropped = [item for sublist in keypoints_cropped for item in sublist]
                    keypoints_cropped = [float(k) for k in keypoints_cropped]
                    keypoints_cropped = torch.tensor(keypoints_cropped)
                    keypoints_cropped = keypoints_cropped.flatten()

                    #Read the file with the original keypoints
                    #They are normalized
                    #They are used to 1)   2) restore good keypoints in the result
                    indexUnderscore = json_file.find('_')
                    json_file = json_file[:indexUnderscore]+".json"#+"_keypoints.json"  
                    original_keypoints_path = join(self.inputpath_original, json_file)
                    if not os.path.isfile(original_keypoints_path):
                    	print("FATAL ERROR: original keypoints path not found: "+original_keypoints_path)
                    	sys.exit()
                    keypoints_original, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(original_keypoints_path, BodyModelOPENPOSE25)
                    keypoints_original, dummy = openPoseUtils.removeConfidence(keypoints_original)
                    keypoints_original = [item for sublist in keypoints_original for item in sublist]
                    keypoints_original = [float(k) for k in keypoints_original]
                    keypoints_original = torch.tensor(keypoints_original)
                    keypoints_original = keypoints_original.flatten()
                    
                    confidence_values = torch.tensor(confidence_values)
                    #print("confidence_values:")
                    #print(confidence_values)
                    yield keypoints_cropped, keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, json_file
                except ValueError as ve:
                	print(ve)
                #except OSError as oe:
                #	print(oe)
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
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 50

# Number of channels in the training images. For color images this is 3
nc = 1

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
  
    datasetModule = eval("datasetBasic")
    dataset = datasetModule.JsonDataset(inputpath_cropped=dataroot_cropped, inputpath_original=dataroot_original, bodyModel=BodyModelOPENPOSE25)

    #dataset = JsonDataset(inputpath_cropped=dataroot_cropped, inputpath_original=dataroot_original)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
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

NEURONS_PER_LAYER_GENERATOR = 512
class Generator(nn.Module):
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
          # First upsampling
          nn.Linear(50+nz, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Second upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Third upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Final upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, 50, bias=False),
          #nn.Tanh()
        )

    def forward(self, batch_of_keypoints_cropped, noise):
        input = torch.cat((batch_of_keypoints_cropped, noise), -1)
        return self.main(input)

# Create the generator
netG_ = modelsECCV18OP.Generator(ngpu, 25)
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
pytorchUtils.explainModel(netG, 1, 1, 28, 28)
#pytorchUtils.computeModel(netG, 1, [{"layer":0, "output":7},{"layer":6, "output":14},{"layer":9, "output":28}])

NEURONS_PER_LAYER_DISCRIMINATOR = 512
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(16,16), stride=2, padding=1, bias=False),
            nn.Linear(50*2, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            #nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, batch_of_keypoints_cropped, batch_of_keypoints_original):
        #print("Discriminator input:",input)
        #print("Discriminator input shape:",input.shape)
        input = torch.cat((batch_of_keypoints_cropped, batch_of_keypoints_original), -1)  
        return self.main(input)

# Create the Discriminator
netD_ = modelsECCV18OP.Discriminator(ngpu, 25)
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
criterion = nn.BCELoss()

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

def testImage(imagePath, keypointsPath):
	#Test over the test image
    keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(keypointsPath, BodyModelOPENPOSE25)
    print("scaleFactor=",scaleFactor)
    keypoints_cropped, confidence_values = openPoseUtils.removeConfidence(keypoints_cropped)
    keypoints_cropped = [item for sublist in keypoints_cropped for item in sublist]
    keypoints_cropped = [float(k) for k in keypoints_cropped]
    keypoints_cropped = torch.tensor(keypoints_cropped)
    confidence_values = torch.tensor(confidence_values)
    keypoints_cropped = keypoints_cropped.flatten()
    print("keypoints_cropped.shape = ", keypoints_cropped.shape)

    batch_of_one_keypoints_cropped = np.reshape(keypoints_cropped, (1, 50))
    batch_of_one_confidence_values = np.reshape(confidence_values, (1, 25))
    fixed_noise_one = torch.randn(1, nz, device=device)

    batch_of_one_keypoints_cropped = batch_of_one_keypoints_cropped.to(device)
    fixed_noise_one = fixed_noise_one.to(device)

    netG.eval()
    fake = netG(batch_of_one_keypoints_cropped, fixed_noise_one).detach().cpu()
    fake = restoreOriginalKeypoints(fake, batch_of_one_keypoints_cropped, batch_of_one_confidence_values)
    netG.train()
    fakeReshapedAsKeypoints = np.reshape(fake, (1, 25, 2))
    fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()

    fakeKeypointsOneImage = fakeReshapedAsKeypoints[0]
    fakeKeypointsOneImage, dummy, dummy, dummy = openPoseUtils.normalize(fakeKeypointsOneImage, BodyModelOPENPOSE25)
    #fakeKeypointsOneImageInt = poseUtils.keypointsToInteger(fakeKeypointsOneImage)

    fakeKeypointsCroppedOneImageIntRescaled = openPoseUtils.denormalize(fakeKeypointsOneImage, scaleFactor, x_displacement, y_displacement)
   	#imgWithKyepoints = np.zeros((500, 500, 3), np.uint8)
    imgWithKyepoints = cv2.imread(imagePath)
    poseUtils.draw_pose(imgWithKyepoints, fakeKeypointsCroppedOneImageIntRescaled, -1, BodyModelOPENPOSE25.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
    cv2.imwrite(OUTPUTPATH+"/test_keypoints.jpg", imgWithKyepoints)

def testMany():
    print('testMany()...')
    batch_of_one_keypoints_cropped = []
    batch_of_one_confidence_values = []
    batch_scaleFactor = []
    batch_x_displacement = []
    batch_y_displacement = []
    batch_filenames = []
    keypointsPath = dataroot_validation
    jsonFiles = [f for f in listdir(keypointsPath) if isfile(join(keypointsPath, f))]
    n = 0
    for filename in jsonFiles:
        #print('Testing '+filename)
        try:
            keypoints_cropped, scaleFactor, x_displacement, y_displacement = openPoseUtils.json2normalizedKeypoints(join(keypointsPath, filename), BodyModelOPENPOSE25)
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
            #print('Skipping '+filename)
            print("WARNING: Cannot draw keypoints ", filename)
            print(e)
            #traceback.print_exc()
            
            

	#batch_of_one_keypoints_cropped = torch.tensor(batch_of_one_keypoints_cropped)
	#batch_of_one_confidence_values = torch.tensor(batch_of_one_confidence_values)
    batch_of_one_keypoints_cropped = torch.stack(batch_of_one_keypoints_cropped)
    batch_of_one_confidence_values = torch.stack(batch_of_one_confidence_values)
    fixed_noise_N = torch.randn(n, nz, device=device)


    batch_of_one_keypoints_cropped = batch_of_one_keypoints_cropped.to(device)
    fixed_noise_N = fixed_noise_N.to(device)
    netG.eval()
    fake = netG(batch_of_one_keypoints_cropped, fixed_noise_N).detach().cpu()
    fake = restoreOriginalKeypoints(fake, batch_of_one_keypoints_cropped, batch_of_one_confidence_values)
    netG.train()
    fakeReshapedAsKeypoints = np.reshape(fake, (n, 25, 2))
    fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
    print("**********************************************")
    print("**********************************************")
    for idx in range(len(fakeReshapedAsKeypoints)):
        fakeKeypointsOneImage = fakeReshapedAsKeypoints[idx]
        fakeKeypointsOneImage, dummy, dummy, dummy = openPoseUtils.normalize(fakeKeypointsOneImage, BodyModelOPENPOSE25)
	   
        fakeKeypointsCroppedOneImageIntRescaled = openPoseUtils.denormalize(fakeKeypointsOneImage, batch_scaleFactor[idx], batch_x_displacement[idx], batch_y_displacement[idx])
        openPoseUtils.keypoints2json(fakeKeypointsCroppedOneImageIntRescaled, OUTPUTPATH+"/Test/keypoints/"+batch_filenames[idx])


        #imgWithKyepoints = np.zeros((500, 500, 3), np.uint8)
        json_file_without_extension = os.path.splitext(batch_filenames[idx])[0]
        json_file_without_extension = json_file_without_extension.replace('_keypoints', '')
        originalImagePath = join(TEST_IMAGES_PATH, json_file_without_extension+".png")
        #print(originalImagePath)
        #imgWithKyepoints = cv2.imread(originalImagePath)
        imgWithKyepoints = pytorchUtils.cv2ReadFile(originalImagePath)
        
        poseUtils.draw_pose(imgWithKyepoints, fakeKeypointsCroppedOneImageIntRescaled, -1, BodyModelOPENPOSE25.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
        try:
            cv2.imwrite(OUTPUTPATH+"/Test/images/"+json_file_without_extension+".jpg", imgWithKyepoints)
            #print("written data/output/Test/"+json_file_without_extension+".jpg")
        except:
            print("WARNING: Cannot find "+originalImagePath)  
        #shutil.copyfile(join("/Users/rtous/DockerVolume/openpose/data/result", json_file_without_extension+"_rendered.png"), join("data/output/Test/"+json_file_without_extension+"_img_cropped_openpose.png"))
    print('testMany() finished.')        

def restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, batch_of_confidence_values):
    '''
    print("DEBUGGING restoreOriginalKeypoints...")
    print("INPUT 1: batch_of_fake_original[0]:")
    print(batch_of_fake_original[0])
    print("INPUT 2: batch_of_keypoints_cropped[0]:")
    print(batch_of_keypoints_cropped[0])
    print("INPUT 3: batch_of_confidence_values[0]:")
    print(batch_of_confidence_values[0])
	'''

    for i, keypoints in enumerate(batch_of_fake_original):
        confidence_values = batch_of_confidence_values[i]
        for c, confidence_value in enumerate(confidence_values):
            if confidence_value > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS:
                #As we work with flat values and keypoints have 2 components we need to do this:
                batch_of_fake_original[i][c*2] = batch_of_keypoints_cropped[i][c*2]
                batch_of_fake_original[i][c*2+1] = batch_of_keypoints_cropped[i][c*2+1]
    '''
    print("OUTPUT: batch_of_fake_original:")
    print(batch_of_fake_original)
    '''
    return batch_of_fake_original
	
tb = SummaryWriter()

print("Starting Training Loop...")
# For each epoch
epoch_idx = 0
MAX_BATCHES = 25000
for epoch in range(num_epochs):        
    print("EPOCH ", epoch)
    # For each batch in the dataloader
    i = 0
    for batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file in dataloader:
        #if i > 1000:
        #	break
        #print("Training iteration "+str(i))

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
        output = netD(batch_of_keypoints_cropped, batch_of_keypoints_original).view(-1)
        #print("Discriminator output: ", output.shape)
        
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        #print("Noise shape: ", noise.shape)
        
        # Generate fake image batch with G
        batch_of_fake_original = netG(batch_of_keypoints_cropped, noise)

        #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
        batch_of_fake_original = restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, confidence_values)

        #As they are fake images let's prepare a batch of labels FAKE
        label.fill_(fake_label)
       
        # Classify all fake batch with D
        output = netD(batch_of_keypoints_cropped, batch_of_fake_original.detach()).view(-1)
        
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        
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
        errG = criterion(output, label)

        # Calculate gradients for G
        errG.backward()
        
        D_G_z2 = output.mean().item()
        
        # Update G
        optimizerG.step()

        tb.add_scalar("Loss", errG.item(), i)

        # Output training stats
        if i % 50 == 0:
            print("**************************************************************")
            print('[%d/%d][%d/?]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, #len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            

            with torch.no_grad():
                fake = netG(batch_of_keypoints_cropped, fixed_noise).detach().cpu()
                
                #We restore the original keypoints (before denormalizing)
                fake = restoreOriginalKeypoints(fake, batch_of_keypoints_cropped, confidence_values)
                

                print("Shape of fake: ", fake.shape)

                fakeReshapedAsKeypoints = np.reshape(fake, (64, 25, 2))
                fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
                
                #print(fakeReshapedAsKeypoints)




                croppedReshapedAsKeypoints = np.reshape(batch_of_keypoints_cropped.cpu(), (64, 25, 2))
                croppedReshapedAsKeypoints = croppedReshapedAsKeypoints.numpy()
 
            #%%capture
        if i % 1000 == 0:  
            NUM_ROWS = 8
            NUM_COLS = 8
            WIDTH = 64
            HEIGHT = 64
            imagesCropped = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
            images = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
            
            ####### DRAW DEBUG POSES FOR THE FIRST 64 IMAGES
            for idx in range(NUM_ROWS*NUM_COLS):
                blank_imageCropped = np.zeros((WIDTH,HEIGHT,3), np.uint8)
                blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)
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
                fakeKeypointsCroppedOneImageInt = fakeKeypointsCroppedOneImage
                fakeKeypointsOneImageInt = fakeKeypointsOneImage
                
                
               	#Draw result over the original image
               	
                
                fakeKeypointsCroppedOneImageIntRescaled = openPoseUtils.denormalize(fakeKeypointsOneImageInt, scaleFactorOneImage, x_displacementOneImage, y_displacementOneImage)
               	
               	#FIX: bad name, they are not normalized!
               	##########
                openPoseUtils.keypoints2json(fakeKeypointsOneImageInt, OUTPUTPATH+"/"+f"{idx:02d}"+"_img_keypoints.json")
                
                json_file_without_extension = os.path.splitext(json_file)[0]
               	json_file_without_extension = json_file_without_extension.replace('_keypoints', '')
               	
               	#Draw the pairs  
                try:
                    poseUtils.draw_pose(blank_imageCropped, fakeKeypointsCroppedOneImageInt, -1, BodyModelOPENPOSE25.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
                    poseUtils.draw_pose(blank_image, fakeKeypointsOneImageInt, -1, BodyModelOPENPOSE25.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
                    targetFilePathCropped = OUTPUTPATH+"/debug_input"+str(idx)+".jpg"
                    targetFilePath = OUTPUTPATH+"/debug"+str(idx)+".jpg"
                    #cv2.imwrite(targetFilePath, blank_image)
                    imagesCropped[int(idx/NUM_COLS)][int(idx%NUM_COLS)] = blank_imageCropped
                    images[int(idx/NUM_COLS)][int(idx%NUM_COLS)] = blank_image
                except Exception:
                    print("WARNING: Cannot draw keypoints ", fakeKeypointsOneImageInt)
                    traceback.print_exc()
            try:
                #print("Assigning: images[int("+str(idx)+"/NUM_COLS)][int("+str(idx)+"%NUM_COLS)]")
                total_imageCropped = poseUtils.concat_tile(imagesCropped)
                total_image = poseUtils.concat_tile(images)
                targetFilePathCropped = OUTPUTPATH+"/debug_input.jpg"
                targetFilePath = OUTPUTPATH+"/debug.jpg"
                cv2.imwrite(targetFilePathCropped, total_imageCropped)
                cv2.imwrite(targetFilePath, total_image)
            except Exception:
                print("WARNING: Cannot draw tile ")
                traceback.print_exc()

            testImage("dynamicData/012.jpg", "dynamicData/012_keypoints.json")
            testMany()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1
        
        i += 1
    print("---- end of epoch "+str(epoch)+"---")
    '''
    if batch_of_keypoints_cropped.size(0) < batch_size:
        print("FATAL ERROR: Batch size = ", batch_of_keypoints_cropped.size(0))
        sys.exit()
    '''
    epoch_idx += 1
tb.close()
print("Finshed. epochs = ", epoch_idx)
