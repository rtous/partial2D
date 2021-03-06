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
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import pyformulas as pf
#from IPython.display import HTML

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
nc = 3

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

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

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
            # input is Z, going into a convolution
            #nn.ConvTranspose2d( in_channels=nz, out_channels=ngf * 8, kernel_size=(4,4), stride=(1,1), padding=(0,0), bias=False),
            nn.ConvTranspose2d( in_channels=nz, out_channels=ngf * 4, kernel_size=(8,8), stride=(1,1), padding=(0,0), bias=False),
            #nn.BatchNorm2d(num_features=ngf * 8),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            
            #nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(ngf * 4),
            #nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            
            #nn.ConvTranspose2d( in_channels=ngf * 4, out_channels=ngf, kernel_size=34, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(ngf),
            nn.ConvTranspose2d( in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d( in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            
            nn.ConvTranspose2d( in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG_ = Generator(ngpu)
netG = netG_.to(device)


############### hook #####################

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
        print(name, output.shape)
    return hook

for i in range(len(netG_.main)):
    netG_.main[i].register_forward_hook(get_activation("Generator layer "+str(i)+" output:"))


########################################

print("layer 0:************************")
conv1 = netG_.main[0]
#Noise shape:  torch.Size([64, 100, 1, 1])
Hin = 1
Win = 1
#Formula ConvTranspose2: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
conv1Hout = (Hin-1)*conv1.stride[0]-2*conv1.padding[0]+conv1.dilation[0]*(conv1.kernel_size[0]-1)+conv1.output_padding[0]+1
conv1Wout = (Win-1)*conv1.stride[1]-2*conv1.padding[1]+conv1.dilation[1]*(conv1.kernel_size[1]-1)+conv1.output_padding[1]+1
print("conv1Hout = ", conv1Hout)
print("conv1Wout = ", conv1Wout)
targetH = 8
kernelSize = ((targetH - (Hin-1)*conv1.stride[0] + 2*conv1.padding[0] - conv1.output_padding[0] - 1)/conv1.dilation[0])+1
print("kernelSize to achieve size "+str(targetH)+" = ", kernelSize)
print("****************************************************")

print("layer 3:*******************")
conv1 = netG_.main[3]
Hin = 8
Win = 8
conv1Hout = int((Hin-1)*conv1.stride[0]-2*conv1.padding[0]+conv1.dilation[0]*(conv1.kernel_size[0]-1)+conv1.output_padding[0]+1)
conv1Wout = int((Win-1)*conv1.stride[1]-2*conv1.padding[1]+conv1.dilation[1]*(conv1.kernel_size[1]-1)+conv1.output_padding[1]+1)
print("conv1Hout = ", conv1Hout)
print("conv1Wout = ", conv1Wout)
targetH = 32
kernelSize = ((targetH - (Hin-1)*conv1.stride[0] + 2*conv1.padding[0] - conv1.output_padding[0] - 1)/conv1.dilation[0])+1
print("kernelSize to achieve size "+str(targetH)+" = ", kernelSize)
print("****************************************************")

print("layer 6:**************")
conv1 = netG_.main[6]
Hin = 32
Win = 32
conv1Hout = int((Hin-1)*conv1.stride[0]-2*conv1.padding[0]+conv1.dilation[0]*(conv1.kernel_size[0]-1)+conv1.output_padding[0]+1)
conv1Wout = int((Win-1)*conv1.stride[1]-2*conv1.padding[1]+conv1.dilation[1]*(conv1.kernel_size[1]-1)+conv1.output_padding[1]+1)
print("conv1Hout = ", conv1Hout)
print("conv1Wout = ", conv1Wout)
targetH = 64
kernelSize = ((targetH - (Hin-1)*conv1.stride[0] + 2*conv1.padding[0] - conv1.output_padding[0] - 1)/conv1.dilation[0])+1
print("kernelSize to achieve size "+str(targetH)+" = ", kernelSize)
print("****************************************************")


# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
print(netD)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

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

        print("Training iteration "+str(i))


        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch

        #print("Length of data: ", len(data))
        #print("Length of data[1]: ", len(data[1]))
        #print(data[1].shape)
        real_cpu = data[0].to(device)
        print("Shape of data[0] [N, C, H, W]: ", real_cpu.shape)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        print("Noise shape: ", noise.shape)
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
        print("errG = criterion(output, label)")
        print("output shape = ", output.shape)
        print("label shape = ", label.shape)
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
                #print("Shape of fake: ", fake.shape)

            
            #%%capture
        if i % 50 == 0:   
            #fig = plt.figure(figsize=(8,8))
            img_list = []
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            plt.axis("off")
            for i in img_list:
              plt.imshow(np.transpose(i,(1,2,0)))
            plt.pause(0.001)
            #plt.savefig('debug.pdf') #Open it with sumatrapdf 
            

            '''
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            for i in img_list:
              plt.imshow(np.transpose(i,(1,2,0)))

            '''

            '''
            now = time.time() - start
            x = np.linspace(now-2, now, 100)
            y = np.sin(2*np.pi*x) + np.sin(3*np.pi*x)
            plt.xlim(now-2,now+1)
            plt.ylim(-3,3)
            plt.plot(x, y, c='black')

        
            fig.canvas.draw()
            image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            screen.update(image)
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
