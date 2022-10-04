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

#BASE CODE: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#CONDITIONAL:
# https://medium.com/analytics-vidhya/step-by-step-implementation-of-conditional-generative-adversarial-networks-54e4b47497d6
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py

#Here it seems to avoid embeddings: https://github.com/togheppi/cDCGAN/blob/master/MNIST_cDCGAN_pytorch.py

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

# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
#batch_size = 128
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

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
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1



########################## MNIST

# Download training data from open datasets.
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
        
        #self.layers = nn.ModuleList()

        self.digit_embedding = nn.Embedding(10, 10)
       
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( in_channels=nz+10, out_channels=ngf * 4, kernel_size=(4,4), stride=(1,1), padding=(0,0), bias=False),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4   

            nn.ConvTranspose2d( in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=(3,3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32         
            
            nn.ConvTranspose2d( in_channels=ngf * 2, out_channels=ngf, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            
            nn.ConvTranspose2d( in_channels=ngf, out_channels=nc, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, noise, digit):
        #print("Generator...")
        #print("digit[0]: ", digit[0].detach().numpy())
        #print("digit: ", digit)
        #print("noise: ", noise)
        #print("self.digit_embedding(digit): ", self.digit_embedding(digit))
        
        gen_input = torch.cat((self.digit_embedding(digit), noise), -1)  
        #print("gen_input shape: ", gen_input.shape)      
        inputReshaped = gen_input.view(b_size, nz+10, 1, 1)
        #print("inputReshaped shape: ", inputReshaped.shape)    
        return self.main(inputReshaped)

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
pytorchUtils.computeModel(netG, 1, [{"layer":0, "output":7},{"layer":6, "output":14},{"layer":9, "output":28}])

'''
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.imageConvolution = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(16,16), stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False),
            #nn.Sigmoid()
            nn.Flatten()

        )
        self.digitConvolution = nn.Sequential(
        	nn.Embedding(10, 10),
        	nn.Flatten(),
        	nn.Linear(10, 10)
        )
        self.finalCombined = nn.Sequential(
        	nn.Linear(522, 1),
        	nn.Sigmoid()
        )

    def forward(self, imageBatch, digitBatch):
        print("Discriminator...")
        imageConvolutionOutput = self.imageConvolution(imageBatch)
        digitConvolutionOutput = self.digitConvolution(digitBatch)
        concatenatedOutput = torch.cat((imageConvolutionOutput, digitConvolutionOutput), 1) #or -1?
        print("concatenatedOutput.shape=",concatenatedOutput.shape)
        return self.finalCombined(concatenatedOutput)
'''
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(16,16), stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Linear(512, 10),
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
netD.apply(weights_init)

# Print the model
print(netD)
#pytorchUtils.explainModel(netD, 28, 28, 1, 1)

# Initialize BCELoss function
criterion = nn.BCELoss()
criterion2 = nn.CrossEntropyLoss(ignore_index=-1)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(batch_size, nz, device=device)
fixed_digits = torch.full((batch_size, ), 7, dtype=int)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.
LABEL_REAL_1 = 1.
LABEL_FAKE_0 = 0.

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
        real_image_batch = data[0].to(device)
        #print("Discriminator REAL input batch (images) = ", real_image_batch)
        #pytorchUtils.showMNIST(np.reshape(real_image_batch[0], 784))
        real_digit_batch = data[1].to(device)
        #print("Discriminator REAL input batch (digits) = ", real_digit_batch)
        #print("Shape of data[0] [N, C, H, W]: ", real_cpu.shape)
        b_size = real_image_batch.size(0)
        batch_of_LABEL_REAL_1 = torch.full((b_size,), LABEL_REAL_1, dtype=torch.float, device=device)
        # Forward pass real batch through D
        DOutput = netD(real_image_batch)
        
        # Calculate loss on all-real batch
        #errD_real = criterion(DOutput, batch_of_LABEL_REAL_1)
        #print("DOutput.shape: ", DOutput.shape)
        #print("real_digit_batch.shape: ", real_digit_batch.shape)
        #print("DOutput: ", DOutput)
        #print("real_digit_batch: ", real_digit_batch)
        errD_real = criterion2(DOutput, real_digit_batch)
        

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = DOutput.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, device=device)
        #random_digits = torch.randint(0, 10, (b_size,))
        #random_digits = torch.randint(low=0, high=10, shape=(b_size, 10, 1, 1), device=device)
        random_digits = torch.randint(10, (b_size, ))
        
        '''
        random_digits = np.expand_dims(random_digits, axis=1)
        random_digits = np.expand_dims(random_digits, axis=1)
        random_digits = torch.tensor(random_digits)
        '''
        #print("random_digits shape", random_digits.shape)
        #random_digits = random_digits.to(device)
        #print("Noise shape: ", noise.shape)

        # Generate fake image batch with G
        fakeImages = netG(noise, random_digits)
        
        ##############
        #print(activation['conv3'])
        
        #batch_of_LABEL_FAKE_0 = torch.full((b_size,), LABEL_FAKE_0, dtype=torch.float, device=device)
        batch_of_LABEL_FAKE_0 = torch.full((b_size,), -1, dtype=torch.long, device=device)

        # Classify all fake batch with D
        Doutput = netD(fakeImages.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion2(Doutput, batch_of_LABEL_FAKE_0)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = Doutput.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        #label.fill_(real_label)  # fake labels are real for generator cost
        #batch_of_LABEL_REAL_1 = torch.full((b_size,), LABEL_REAL_1, dtype=torch.float, device=device)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        Doutput = netD(fakeImages)
        # Calculate G's loss based on this output
        errG = criterion2(Doutput, random_digits)
        #print("errG = criterion(output, label)")
        #print("output shape = ", output.shape)
        #print("label shape = ", label.shape)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = Doutput.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_digits).detach().cpu()
                print("Shape of fake: ", fake.shape)

            
            #%%capture
        if i % 50 == 0:   
            #fig = plt.figure(figsize=(8,8))
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
