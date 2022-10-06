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

#in PyTorch the format is "Batch Size x Channel x Height x Width"
# (al revés que a Tesnsorflow)

CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS = 0.1 

#   size using a transformer.
#image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 15

# Size of z latent vector (i.e. size of generator input)
nz = 15

# Size of feature maps in generator
#ngf = 64
ngf = 16

# Size of feature maps in discriminator
#ndf = 64
ndf = 4#16

NEURONS_PER_LAYER_GENERATOR = 32
'''
class Generator(nn.Module):
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( in_channels=nz, out_channels=ngf * 4, kernel_size=(4,4), stride=(1,1), padding=(0,0), bias=False),
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
        )

    def forward(self, batch_of_keypoints_cropped, noise):
        #input = torch.cat((batch_of_keypoints_cropped, noise), -1)
        input = batch_of_keypoints_cropped
        return self.main(input)
'''
class Generator(nn.Module):
    def __init__(self, channels=nc):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True, relu=True):
            layers = [nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if relu:
                layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True, relu=True):
            layers = [nn.ConvTranspose2d(in_channels=in_feat, out_channels=out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if relu:
                layers.append(nn.ReLU())
            return layers
        self.model = nn.Sequential(
            *downsample(channels, 8, normalize=False),
            *downsample(8, 16),
            *downsample(16, 100, normalize=False, relu=False),
            #nn.Conv2d(128, 1000, kernel_size=4, stride=2, padding=1),
            *upsample(100, 16),
            *upsample(16, 8),
            *upsample(8, channels, normalize=False, relu=False),
            #nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        '''
        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 128),
            *downsample(128, 1000, normalize=False, relu=False),
            #nn.Conv2d(128, 1000, kernel_size=4, stride=2, padding=1),
            *upsample(1000, 128),
            *upsample(128, 64),
            *upsample(64, channels, normalize=False, relu=False),
            #nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        '''

    def forward(self, x):
        #print("Generator received: ", x.shape)
        #print("Generator...")
        return self.model(x)

'''
NEURONS_PER_LAYER_DISCRIMINATOR = 16
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc * 2, out_channels=ndf, kernel_size=(16,16), stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=(8,8), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=(8,8), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=ndf * 4, out_channels=1, kernel_size=(2,2), stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )
'''
#NEURONS_PER_LAYER_DISCRIMINATOR = 4
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc * 2, out_channels=ndf, kernel_size=(8,8), stride=8, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=(4,4), stride=4, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=(8,8), stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=ndf * 2, out_channels=1, kernel_size=(2,2), stride=2, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, batch_of_keypoints_cropped, batch_of_keypoints_original):
        #print("Discriminator input shape befor concat: batch_of_keypoints_cropped.shape=",batch_of_keypoints_cropped.shape)
        #print("Discriminator input shape befor concat: batch_of_keypoints_original.shape=",batch_of_keypoints_original.shape)
        
        #Mirar això: https://www.tensorflow.org/tutorials/generative/pix2pix
        input = torch.cat((batch_of_keypoints_cropped, batch_of_keypoints_original), -3)  
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
