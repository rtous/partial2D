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

# Root directory for dataset
dataroot = "dynamicData/H36M_ECCV18_FILTERED"
OUTPUTPATH = "data/output"
pathlib.Path(OUTPUTPATH).mkdir(parents=True, exist_ok=True) 

#To avoid parallel error on macos (change for )
# Number of workers for dataloader
workers = 2
#Also run export OMP_NUM_THREADS=1 in the terminal

class JsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, inputpath):
        self.inputpath = inputpath
        self.jsonFiles = [f for f in listdir(self.inputpath) if isfile(join(self.inputpath, f)) and f.endswith("json") ]
    
    def __iter__(self):
        #jsonFiles = [f for f in listdir(self.inputpath) if isfile(join(self.inputpath, f)) and f.endswith("json") ]
        for json_file in self.jsonFiles:
            try:
                keypoints = openPoseUtils.json2normalizedKeypoints(join(self.inputpath, json_file))
                keypoints = openPoseUtils.removeConfidence(keypoints)
                keypoints = [item for sublist in keypoints for item in sublist]
                keypoints = [float(k) for k in keypoints]
                #keypoints = np.array(keypoints)
                #keypoints = keypoints.flatten()
                #print("keypoints yield = ", keypoints)
                #keypoints = keypoints.astype(np.float)
                keypoints = torch.tensor(keypoints)
                keypoints = keypoints.flatten()
                '''
                keypoints = torch.tensor([ 
             -70.,   38.,   27.,   33.,   33.,  -75.,   26.,  -69.,   29.,   36.,
             -29.,   27.,  -26.,  -37.,   37.,   36.,   41.,   33.,   31.,   34.,
              24., -155.,   35.,   38.,  -37.,   38.,   37.,  -51.,  -83.,   26.,
              26.,   28.,   28., -115.,  -27.,   55.,   34.,   32.,   27.,   26.,
              52.,   25.,   26.,   29.,   28.,  -47.,   37.,   31.,  -57.,   38.
              ])'''
                #keypoints = np.array(keypoints)
                yield keypoints
            except:
                print("WARNING: Error reading ", json_file)

    def __len__(self):
        return len(self.jsonFiles)
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
batch_size = 32

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

########################## MNIST

# Download training data from open datasets.
'''
training_data = dset.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
batch_size = 64

# Create data loaders.
dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
'''
def prepare_dataset():
  
    '''
    dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    '''
    dataset = JsonDataset(inputpath=dataroot)

    

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=workers)

    # Batch and shuffle data with DataLoader
    #trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # Return dataset through DataLoader
    return dataloader

dataloader = prepare_dataset()

#pytorchUtils.explainDataloader(dataloader)
##########################


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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
'''
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution

            #nn.ConvTranspose2d( in_channels=nz, out_channels=ngf * 4, kernel_size=(4,4), stride=(1,1), padding=(0,0), bias=False),
            nn.Linear(nz, 128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4   

            #nn.ConvTranspose2d( in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=(3,3), stride=2, padding=1, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32         
            
            #nn.ConvTranspose2d( in_channels=ngf * 2, out_channels=ngf, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            
            #nn.ConvTranspose2d( in_channels=ngf, out_channels=nc, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.Linear(128, 50, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        print("Generator input shape: ",input.shape)
        #inputReshaped = input.view(b_size, nz, 1, 1)
        #print("inputReshaped shape: ", inputReshaped.shape) 
        return self.main(input)
'''
NEURONS_PER_LAYER_GENERATOR = 512
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
          # First upsampling
          nn.Linear(nz, NEURONS_PER_LAYER_GENERATOR, bias=False),
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

    def forward(self, input):
        return self.main(input)

# Create the generator
netG_ = Generator(ngpu)
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
            nn.Linear(50, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
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

    def forward(self, input):
        #print("Discriminator input:",input)
        #print("Discriminator input shape:",input.shape)
        return self.main(input)


# Create the Discriminator
netD_ = Discriminator(ngpu)
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
fixed_noise = torch.randn(64, nz, device=device)

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

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):


        #print("Data received:", data[0])

        #print("Training iteration "+str(i))

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch

        #print("Length of data: ", len(data))
        #print("Length of data[1]: ", len(data[1]))
        #print(data[1].shape)
        
        real_cpu = data.to(device)

        #real_cpu = torch.tensor(data, device=device)


        #print("real_cpu = ", real_cpu)
        #print("Shape of real_cpu [N, C, H, W]: ", real_cpu.shape)
        #print("dtype of real_cpu: ", real_cpu.dtype)
        #print("Shape of data[0] [N, C, H, W]: ", real_cpu.shape)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        
        #real_cpu = torch.randn(b_size, 50, device=device)#BORRAR!
        #print(real_cpu.dtype)
        output = netD(real_cpu).view(-1)
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
        fake = netG(noise)
        
        ##############
        #print(activation['conv3'])
        
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
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
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        #print("errG = criterion(output, label)")
        #print("output shape = ", output.shape)
        #print("label shape = ", label.shape)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print("**************************************************************")
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                print("Shape of fake: ", fake.shape)

                fakeReshapedAsKeypoints = np.reshape(fake, (64, 25, 2))
                fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
                #print(fakeReshapedAsKeypoints)

            
            #%%capture
        if i % 1000 == 0:  
            NUM_ROWS = 8
            NUM_COLS = 8
            WIDTH = 64
            HEIGHT = 64
            images = np.empty(shape=(NUM_ROWS, NUM_COLS),dtype='object')
            


            
            for idx in range(NUM_ROWS*NUM_COLS):
                blank_image = np.zeros((WIDTH,HEIGHT,3), np.uint8)
                fakeKeypointsOneImage = fakeReshapedAsKeypoints[idx]
                #print("fakeKeypointsOneImage:", fakeKeypointsOneImage)
                fakeKeypointsOneImage = openPoseUtils.normalize(fakeKeypointsOneImage)
                #print("normalizedFakeKeypointsOneImage:", fakeKeypointsOneImage)
                fakeKeypointsOneImageInt = poseUtils.keypointsToInteger(fakeKeypointsOneImage)
                #print("integer normalizedFakeKeypointsOneImage:", fakeKeypointsOneImageInt)
                #print("Trying to draw:", fakeKeypointsOneImageInt)
                #poseUtils.draw_pose(blank_image, fakeKeypointsOneImageInt, -1, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
               	
               	openPoseUtils.normalizedKeypoints2json(fakeKeypointsOneImageInt, "data/output/"+str(idx)+".json")

                try:
                    #print("Drawing fakeKeypointsOneImage:")
                    #print(fakeKeypointsOneImageList)
                    poseUtils.draw_pose(blank_image, fakeKeypointsOneImageInt, -1, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
                    targetFilePath = "data/output/debug"+str(idx)+".jpg"
                    #cv2.imwrite(targetFilePath, blank_image)
                    images[int(idx/NUM_COLS)][int(idx%NUM_COLS)] = blank_image
                except Exception:
                    print("WARNING: Cannot draw keypoints ", fakeKeypointsOneImageInt)
                    traceback.print_exc()
                
            
            try:
                
                #print("Assigning: images[int("+str(idx)+"/NUM_COLS)][int("+str(idx)+"%NUM_COLS)]")
                total_image = poseUtils.concat_tile(images)
                targetFilePath = "data/output/debug.jpg"
                cv2.imwrite(targetFilePath, total_image)
            except Exception:
                print("WARNING: Cannot draw tile ")
                traceback.print_exc()
            

# Closing
            #poseUtils.draw_pose(blank_image, keypoints, -1, openPoseUtils.POSE_BODY_25_PAIRS_RENDER_GP, openPoseUtils.POSE_BODY_25_COLORS_RENDER_GPU, False)
            #targetFilePath = join(OUTPUTPATH, filename_noextension+".jpg")
            #print("Writing image to "+targetFilePath)
            #cv2.imwrite(targetFilePath, blank_image)

            '''
            img_list = []
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            plt.axis("off")
            for i in img_list:
              print("Shape of i: ", i.shape)
              i_transposed = np.transpose(i,(1,2,0))
              print("Shape of i_transposed: ", i_transposed.shape)
              plt.imshow(i_transposed)
            if arguments.interactive:
              plt.pause(0.001)
            else:
              plt.savefig(OUTPUTPATH+'/debug.pdf') #Open it with sumatrapdf 
            '''
            
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        '''
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                print("Shape of fake: ", fake.shape)
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            #%%capture
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            for i in img_list:
              plt.imshow(np.transpose(i,(1,2,0)))
            plt.show()
        '''

        iters += 1
