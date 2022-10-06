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

#pix2pix: https://github.com/znxlwm/pytorch-pix2pix/blob/master/network.py


#in PyTorch the format is "Batch Size x Channel x Height x Width"
# (al revés que a Tesnsorflow)

CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS = 0.1 

#   size using a transformer.
#image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 15

# Size of z latent vector (i.e. size of generator input)
#nz = 1000

# Size of feature maps in generator
#ngf = 64
ngf = 16

# Size of feature maps in discriminator
#ndf = 64
ndf = 4#16

NEURONS_PER_LAYER_GENERATOR = 32

def noiseSquareBatch(outputRes, batchsize):
    #noiseSquareBatch = np.zeros((batchsize, 1, outputRes, outputRes), dtype="float32")
    noiseSquareBatch = np.random.random_sample((batchsize, 1, outputRes, outputRes)).astype('float32')
    return noiseSquareBatch

class Generator64(nn.Module):
#v1 (64 pixels)
    def __init__(self, channels, nz, addNoise=False):
        super(Generator64, self).__init__()
        self.nc = channels
        self.addNoise = addNoise
        self.nz = nz
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


#FOR 128 pixels
class Generator128(nn.Module):
    def __init__(self, channels, nz, addNoise=False):
        super(Generator128, self).__init__()
        self.nc = channels
        self.addNoise = addNoise
        self.nz = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            #nn.ConvTranspose2d( in_channels=nz, out_channels=ngf * 8, kernel_size=(4,4), stride=(1,1), padding=(0,0), bias=False),
            nn.ConvTranspose2d( in_channels=nz, out_channels=ngf * 8, kernel_size=8, stride=2, padding=0, bias=False),
            #nn.BatchNorm2d(num_features=ngf * 8),
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
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


#For 64 pixels
class Discriminator64(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator64, self).__init__()
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

    def forward(self, batch_of_keypoints_original):
        #print("Discriminator input shape befor concat: batch_of_keypoints_cropped.shape=",batch_of_keypoints_cropped.shape)
        #print("Discriminator input shape befor concat: batch_of_keypoints_original.shape=",batch_of_keypoints_original.shape)
        
        #Mirar això: https://www.tensorflow.org/tutorials/generative/pix2pix
        #input = torch.cat((batch_of_keypoints_cropped, batch_of_keypoints_original), -3)  
        
        input = batch_of_keypoints_original
        return self.main(input)

#128 pixels?
class Discriminator128(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator128, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(8,8), stride=4, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=(2,2), stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, batch_of_keypoints_original):
        #print("Discriminator input shape befor concat: batch_of_keypoints_cropped.shape=",batch_of_keypoints_cropped.shape)
        #print("Discriminator input shape befor concat: batch_of_keypoints_original.shape=",batch_of_keypoints_original.shape)
        
        #Mirar això: https://www.tensorflow.org/tutorials/generative/pix2pix
        #input = torch.cat((batch_of_keypoints_cropped, batch_of_keypoints_original), -3)  
        
        input = batch_of_keypoints_original
        #print("Discriminator input shape:",input.shape)
        # Hauria de ser (batch_size, 128, 128, channels*2)
        #print("Discriminator input[0] after concat:",input[0][0])
        #print("Discriminator...")
        return self.main(input)

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
    
    #batch_of_fake_original_ = batch_of_fake_original.copy()
    batch_of_fake_original_ = batch_of_fake_original.detach()

    for i, keypoints in enumerate(batch_of_fake_original):
        #if i == 0:
        #    print("received batch_of_fake_original[i]: ", keypoints)
        #    print("received batch_of_keypoints_cropped[i]: ", batch_of_keypoints_cropped[i])
        #    print("received batch_of_confidence_values[i]: ", batch_of_confidence_values[i])
        confidence_values = batch_of_confidence_values[i]
        #print("len(confidence_values)=",len(confidence_values))
        #print("batch_of_keypoints_cropped[i](inner):", batch_of_keypoints_cropped[i])
        #print("fake before restoring(inner):", keypoints)
        for c, confidence_value in enumerate(confidence_values):
            if confidence_value > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS:
                #As we work with flat values and keypoints have 2 components we need to do this:
                #print("Restoring idx "+str(c)+" with confidence_value "+str(confidence_value))
                
                #batch_of_fake_original[i][c*2] = batch_of_keypoints_cropped[i][c*2]
                #batch_of_fake_original[i][c*2+1] = batch_of_keypoints_cropped[i][c*2+1]
                
                batch_of_fake_original_[i, c, :, :] = batch_of_keypoints_cropped[i, c, :, :]

        #if i == 0:
        #    print("resulting keypoints: ", batch_of_fake_original[i])
        #print("fake after restoring(inner):", keypoints)
    '''
    print("OUTPUT: batch_of_fake_original:")
    print(batch_of_fake_original)
    '''
    return batch_of_fake_original_
