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
import train_utils
import numpy as np
from torch.nn import functional as F




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

class Models():
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, ngpu, numJoints, nz, KEYPOINT_RESTORATION, device = torch.device("cpu")):
        super(Models, self).__init__()
        #generator
        #netG_ = Generator(ngpu, numJoints, nz)
        netG_ = VAE(numJoints*2, nz).to(device)

        netG = netG_.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))
        netG.apply(train_utils.weights_init)
        print(netG)

        self.netG = netG
        self.KEYPOINT_RESTORATION = KEYPOINT_RESTORATION



def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, numJoints*2), reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    #torch.nn.MSELoss()
    BCE = F.mse_loss(recon_x, x, size_average=None, reduce=None, reduction='sum')#mean

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
    #return 10000*(BCE + KLD)
    #Here suggests to change this: https://www.quora.com/How-do-you-fix-a-Variational-Autoencoder-VAE-that-suffers-from-mode-collapse


class TrainSetup():
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, models, ngpu, numJoints, nz, lr, beta1, PIXELLOSS_WEIGHT, device = "cpu"):
        super(TrainSetup, self).__init__()

        #loss function
        self.lossFunctionD = nn.BCELoss() #torch.nn.BCEWithLogitsLoss
        self.lossFunctionG_adversarial = nn.BCELoss() 
        self.lossFunctionG_regression = torch.nn.MSELoss()#torch.nn.MSELoss() #torch.nn.L1Loss()

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        #self.optimizerG = optim.Adam(models.netG.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(models.netG.parameters(), lr=1e-3)
        self.nz = nz


def trainStep(models, trainSetup, b_size, device, tb, step_absolute, num_epochs, epoch, i, batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file):
    
    models.netG.zero_grad()
    
    # Generate batch of latent vectors
    noise = torch.randn(b_size, trainSetup.nz, device=device)
    
    #CVAE code
    recon_batch, mu, logvar = models.netG(batch_of_keypoints_original)
    
    trainSetup.optimizerG.zero_grad()

    errG = loss_function(recon_batch, batch_of_keypoints_original, mu, logvar)

    errG.backward()
    
    trainSetup.optimizerG.step()

    tb.add_scalar("LossG", errG.item(), step_absolute)

    # Output training stats each 50 batches
        
    if i % 50 == 0:
        print("**************************************************************")
        print('[%d/%d][%d/?]\tLoss_G: %.4f'
              % (epoch, num_epochs, i, errG.item()))

def inference(models, b_size, noise, numJoints, batch_of_keypoints_cropped, confidence_values):
    with torch.no_grad():
        fake = models.netG.decode(noise).detach().cpu()

        #We restore the original keypoints (before denormalizing)
        if models.KEYPOINT_RESTORATION:
            fake = restoreOriginalKeypoints(fake, batch_of_keypoints_cropped, confidence_values)
        print("Shape of fake: ", fake.shape)
        fakeReshapedAsKeypoints = np.reshape(fake, (b_size, numJoints, 2))
        fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
        
        return fakeReshapedAsKeypoints

def save(models, OUTPUTPATH, epoch, i):
    torch.save(models.netG.state_dict(), OUTPUTPATH+"/model/model_epoch"+str(epoch)+"_batch"+str(i)+".pt")

def load(models, MODELPATH):
    models.netG.load_state_dict(torch.load(MODELPATH))