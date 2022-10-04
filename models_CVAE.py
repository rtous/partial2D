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
class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        print("feature_size + class_size", feature_size + class_size)

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, NEURONS_PER_LAYER_CVAE)
        #self.bn1 = nn.BatchNorm1d(400, 0.8)
           
        #self.fc2  = nn.Linear(400, 400)
        #self.bn2 = nn.BatchNorm1d(400, 0.8)

        self.fc21 = nn.Linear(NEURONS_PER_LAYER_CVAE, latent_size)
        self.fc22 = nn.Linear(NEURONS_PER_LAYER_CVAE, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, NEURONS_PER_LAYER_CVAE)
        self.fc4 = nn.Linear(NEURONS_PER_LAYER_CVAE, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
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

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        #return self.sigmoid(self.fc4(h3))
        return self.fc4(h3) #our (normalized) data is not in the range 0,1

    def forward(self, x, c):
        '''
        print(" --- forward --- ")
        print("x.shape = ", x.shape)
        print("x[0] = ", x[0])
        print("c.shape = ", c.shape)
        print("c[1] = ", c[1])
        '''
        #mu, logvar = self.encode(x.view(-1, 28*28), c)
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar



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

def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, numJoints*2), reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE = F.mse_loss(recon_x, x, size_average=None, reduce=None, reduction='sum')#mean


    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class Models():
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, ngpu, numJoints, nz, KEYPOINT_RESTORATION, device = torch.device("cpu")):
        super(Models, self).__init__()
        #generator
        #netG_ = Generator(ngpu, numJoints, nz)

        netG_ = CVAE(numJoints*2, nz, numJoints*2)


        netG = netG_.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))
        netG.apply(train_utils.weights_init)
        print(netG)

        #pytorchUtils.explainModel(netD, 28, 28, 1, 1)
        #if arguments.debug:
        #    pytorchUtils.registerDebugHook(netD_)

        self.netG = netG
        
        self.KEYPOINT_RESTORATION = KEYPOINT_RESTORATION



class TrainSetup():
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, models, ngpu, numJoints, nz, lr, beta1, PIXELLOSS_WEIGHT, device = "cpu"):
        super(TrainSetup, self).__init__()

        #loss function
        self.lossFunctionG_adversarial = nn.BCELoss() 
        self.lossFunctionG_regression = torch.nn.MSELoss()#torch.nn.MSELoss() #torch.nn.L1Loss()

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        self.optimizerG = optim.Adam(models.netG.parameters(), lr=1e-3)

        self.nz = nz


def trainStep(models, trainSetup, b_size, device, tb, step_absolute, num_epochs, epoch, i, batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file):
    #Batch of real labels
    label = torch.full((b_size,), trainSetup.real_label, dtype=torch.float, device=device)

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, trainSetup.nz, device=device)
    
    # Generate fake image batch with G
    batch_of_fake_original, mu, logvar = models.netG(batch_of_keypoints_original, batch_of_keypoints_cropped)

    #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
    if models.KEYPOINT_RESTORATION:
        batch_of_fake_original = restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, confidence_values)

    models.netG.zero_grad()
        
    errG = loss_function(batch_of_fake_original, batch_of_keypoints_original, mu, logvar)

    errG.backward()
    
    trainSetup.optimizerG.step()

    tb.add_scalar("LossG", errG.item(), step_absolute)

    # Output training stats each 50 batches
        
    if i % 50 == 0:
        print("**************************************************************")
        print('[%d/%d][%d/?]\tLoss_G: %.4f'
              % (epoch, num_epochs, i, errG.item()))

def inference(models, b_size, noise, numJoints, batch_of_keypoints_cropped, confidence_values):
    #returns normalized and flat values
    with torch.no_grad():
        print("drawing batch...")
        
        fake = models.netG.decode(noise, batch_of_keypoints_cropped).detach().cpu()

        #We restore the original keypoints (before denormalizing)
        if models.KEYPOINT_RESTORATION:
            fake = restoreOriginalKeypoints(fake, batch_of_keypoints_cropped, confidence_values)
        print("Shape of fake: ", fake.shape)
        #fakeReshapedAsKeypoints = np.reshape(fake, (b_size, numJoints, 2))
        #fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
        
        return fake

def save(models, OUTPUTPATH, epoch, i):
    torch.save(models.netG.state_dict(), OUTPUTPATH+"/model/model_epoch"+str(epoch)+"_batch"+str(i)+".pt")

def load(models, MODELPATH):
    models.netG.load_state_dict(torch.load(MODELPATH))