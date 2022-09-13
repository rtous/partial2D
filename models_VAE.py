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

CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS = 0.1 

NEURONS_PER_LAYER_CVAE = 64
class VAE(nn.Module):
    def __init__(self, feature_size, latent_size):
        super(VAE, self).__init__()
        self.feature_size = feature_size
        #print("feature_size + class_size", feature_size + class_size)

        # encode
        self.fc1  = nn.Linear(feature_size, NEURONS_PER_LAYER_CVAE)
        #self.bn1 = nn.BatchNorm1d(NEURONS_PER_LAYER_CVAE, 0.8)
           
        #self.fc2  = nn.Linear(NEURONS_PER_LAYER_CVAE, NEURONS_PER_LAYER_CVAE)
        #self.bn2 = nn.BatchNorm1d(NEURONS_PER_LAYER_CVAE, 0.8)

        self.fc21 = nn.Linear(NEURONS_PER_LAYER_CVAE, latent_size)
        self.fc22 = nn.Linear(NEURONS_PER_LAYER_CVAE, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size, NEURONS_PER_LAYER_CVAE)
        self.fc4 = nn.Linear(NEURONS_PER_LAYER_CVAE, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = x # (bs, feature_size+class_size)
        #print("inputs.shape: ", inputs.shape)
        h1 = self.elu(self.fc1(inputs))
        #h1 = self.elu(self.bn1(self.fc1(inputs)))
        #h2 = self.elu(self.bn2(self.fc2(h1)))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = z # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        #return self.sigmoid(self.fc4(h3))
        return self.fc4(h3)

    def forward(self, x):
        '''
        print(" --- forward --- ")
        print("x.shape = ", x.shape)
        print("x[0] = ", x[0])
        print("c.shape = ", c.shape)
        print("c[1] = ", c[1])
        '''
        #mu, logvar = self.encode(x.view(-1, 28*28), c)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



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
