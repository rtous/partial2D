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


'''
#Base code is from Pytorch tutorial on GAN for faces but
#(not anymore)The network is tuned as here: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
#This one: https://www.machinecurve.com/index.php/2021/07/17/building-a-simple-vanilla-gan-with-pytorch/

#He tingut que canviar el shape del input (la imatge also el soroll)
#Expects flat input

#CHANGE 1 SHAPES

0) Data loader:
torch.Size([128, 1, 28, 28])
torch.Size([128]) torch.int64

1) Real images
batch_of_real_images = data[0].view(data[0].size(0), -1)
#batch_of_real_images = data[0]
->Discriminator receives a real batch of shape  torch.Size([128, 784])

2) Noise
#noise = torch.randn(b_size, nz, 1, 1, device=device)
noise = torch.randn(b_size, nz, device=device)
->noise shape =  torch.Size([128, 100])

CHANGE 2) BATCH SIZE = 128

CHANGE 3) manualSeed = 42

CHANGE 4) netD/netG.apply(weights_init) ho he tret

'''



parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default = 0)
parser.add_argument('--interactive', type=int, default = 0)
arguments, unparsed = parser.parse_known_args()

# Set random seed for reproducibility
manualSeed = 42


#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128
#batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
#ngf = 64
ngf = 64

# Size of feature maps in discriminator
#ndf = 64
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

GENERATOR_OUTPUT_IMAGE_SHAPE = image_size * image_size

########################## MNIST
'''
# Download training data from open datasets.
training_data = dset.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Create data loaders.
dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
'''
def prepare_dataset():
  
  """ Prepare dataset through DataLoader """
  # Prepare MNIST dataset
  dataset = MNIST(os.getcwd(), download=True, train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ]))
  # Batch and shuffle data with DataLoader
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
  # Return dataset through DataLoader
  return trainloader
dataloader = prepare_dataset()



pytorchUtils.explainDataloader(dataloader)
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


class Generator(nn.Module):
	def __init__(self, ngpu):
		super(Generator, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
          # First upsampling
          nn.Linear(nz, 128, bias=False),
          nn.BatchNorm1d(128, 0.8),
          nn.LeakyReLU(0.25),
          # Second upsampling
          nn.Linear(128, 256, bias=False),
          nn.BatchNorm1d(256, 0.8),
          nn.LeakyReLU(0.25),
          # Third upsampling
          nn.Linear(256, 512, bias=False),
          nn.BatchNorm1d(512, 0.8),
          nn.LeakyReLU(0.25),
          # Final upsampling
          nn.Linear(512, GENERATOR_OUTPUT_IMAGE_SHAPE, bias=False),
          nn.Tanh()
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
#netG.apply(weights_init)

# Print the model
print(netG)
#pytorchUtils.explainModel(netG, 1, 1, 28, 28)
#pytorchUtils.computeModel(netG, 1, [{"layer":0, "output":7},{"layer":6, "output":14},{"layer":9, "output":28}])

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(GENERATOR_OUTPUT_IMAGE_SHAPE, 1024), 
            nn.LeakyReLU(0.25),
            nn.Linear(1024, 512), 
            nn.LeakyReLU(0.25),
            nn.Linear(512, 256), 
            nn.LeakyReLU(0.25),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
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
#netD.apply(weights_init)

# Print the model
print(netD)
#pytorchUtils.explainModel(netD, 28, 28, 1, 1)
#pytorchUtils.computeModel(netD, 28, [{"layer":0, "output":14},{"layer":2, "output":7},{"layer":5, "output":4},{"layer":8, "output":1}])

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
#fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
fixed_noise = torch.randn(batch_size, nz, device=device)
print("noise shape = ", fixed_noise.shape)

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

        batch_of_real_images = data[0].view(data[0].size(0), -1)
        #batch_of_real_images = data[0]

        real_cpu = batch_of_real_images.to(device)
        #print("Shape of data[0] [N, C, H, W]: ", real_cpu.shape)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D

        if arguments.debug:
        	print("Discriminator receives a real batch of shape ", real_cpu.shape) 
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        #noise = torch.randn(b_size, nz, 1, 1, device=device)

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
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                print("Shape of fake: ", fake.shape)

            
            #%%capture
        if i % 50 == 0:   
            #fig = plt.figure(figsize=(8,8))
            '''
            fakes_reshaped = np.empty([batch_size, 28, 28])
            i = 0
            for img in fake:
            	img_reshaped = np.reshape(img, (28, 28))
            	fakes_reshaped[i] = img_reshaped #RUBEN
            	i = i + 1
			'''
            #fakes_reshaped = np.reshape(fake, (batch_size, 28, 28))

            fake2 = np.expand_dims(fake, axis=1)
            print("Shape of fake2: ", fake2.shape)
            fake3 = torch.from_numpy(np.reshape(fake2, newshape=(128, 1, 28, 28)))
            print("Shape of fake3: ", fake3.shape)

            #fakes_reshaped = np.reshape(fake, (batch_size, 28, 28))
            img_list = []
            img_list.append(vutils.make_grid(fake3, padding=2, normalize=True))
            plt.axis("off")
            for i in img_list:
               #i_reshaped = np.reshape(i, (28, 28))
               #plt.imshow(i)
               plt.imshow(np.transpose(i,(1,2,0)))
            if arguments.interactive:
              plt.pause(0.001)
            else:
              plt.savefig('debug.pdf') #Open it with sumatrapdf 
            
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
