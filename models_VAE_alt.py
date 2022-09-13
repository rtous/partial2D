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

from torch.nn import functional as F


CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS = 0.1 

#   size using a transformer.
#numjoints = 15#25


# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
#nz = 100

# Size of feature maps in generator
#ngf = 64
ngf = 16

# Size of feature maps in discriminator
#ndf = 64
ndf = 16

NEURONS_PER_LAYER_GENERATOR = 256
NEURONS_PER_LAYER_GENERATOR_EMBEDDING = 128
class Generator(nn.Module):
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, ngpu, numJoints):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.numJoints = numJoints
        self.image_size = self.numJoints*2
        self.main = nn.Sequential(
          # First upsampling
          nn.Linear(self.image_size+nz, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Second upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, NEURONS_PER_LAYER_GENERATOR_EMBEDDING, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR_EMBEDDING, 0.8),
          nn.LeakyReLU(0.25),
          # Third upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR_EMBEDDING, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Final upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, self.image_size, bias=False),
          #nn.Tanh() # result in [-1,1] 
        )

    def forward(self, batch_of_keypoints_cropped, noise):
        input = torch.cat((batch_of_keypoints_cropped, noise), -1)
        return self.main(input)

NEURONS_PER_LAYER_DISCRIMINATOR = 256
class Discriminator(nn.Module):
    def __init__(self, ngpu, numJoints):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.numJoints = numJoints
        self.image_size = self.numJoints*2
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(16,16), stride=2, padding=1, bias=False),
            nn.Linear(self.image_size*2, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            nn.BatchNorm1d(NEURONS_PER_LAYER_DISCRIMINATOR, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            #nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            nn.BatchNorm1d(NEURONS_PER_LAYER_DISCRIMINATOR, 0.8),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            nn.BatchNorm1d(NEURONS_PER_LAYER_DISCRIMINATOR, 0.8),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, 1, bias=False),
            nn.Sigmoid() #result in [0,1]
            #idea: supress sigmoid from https://github.com/soumith/ganhacks/issues/36
        )

    def forward(self, batch_of_keypoints_cropped, batch_of_keypoints_original): 
        #print("Discriminator batch_of_keypoints_cropped shape:",batch_of_keypoints_cropped.shape)
        #print("Discriminator batch_of_keypoints_original shape:",batch_of_keypoints_original.shape)
        input = torch.cat((batch_of_keypoints_cropped, batch_of_keypoints_original), -1)  
        #print("Discriminator input:",input.shape)
        return self.main(input)

NEURONS_PER_LAYER_CVAE = 128
class VAE(nn.Module):
    def __init__(self, feature_size, latent_size):
        super(VAE, self).__init__()
        self.feature_size = feature_size
        self.latent_size = latent_size
    
        # encoder
        self.enc1 = nn.Linear(in_features=feature_size, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=latent_size*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=latent_size, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=feature_size)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, self.latent_size)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var



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

    batch_of_fake_original = batch_of_fake_original.clone()

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
                batch_of_fake_original[i][c*2] = batch_of_keypoints_cropped[i][c*2]
                batch_of_fake_original[i][c*2+1] = batch_of_keypoints_cropped[i][c*2+1]
        #if i == 0:
        #    print("resulting keypoints: ", batch_of_fake_original[i])
        #print("fake after restoring(inner):", keypoints)
    '''
    print("OUTPUT: batch_of_fake_original:")
    print(batch_of_fake_original)
    '''
    return batch_of_fake_original
