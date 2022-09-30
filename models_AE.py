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

CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS = 0.1 

NEURONS_PER_LAYER_GENERATOR = 256
NEURONS_PER_LAYER_GENERATOR_EMBEDDING = 128
BIAS = False
class Generator(nn.Module):
    def __init__(self, ngpu, numJoints, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.numJoints = numJoints
        self.image_size = self.numJoints*2
        self.nz = nz
        self.encoder = nn.Sequential(
          # First upsampling
          nn.Linear(self.image_size, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Second upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, nz, bias=False),
          nn.BatchNorm1d(nz, 0.8),
          nn.LeakyReLU(0.25),
        )
        self.decoder = nn.Sequential(
          # Third upsampling
          nn.Linear(nz, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Final upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, self.image_size, bias=False),
        )

    def encode(self, batch_of_keypoints_cropped): # Q(z|x, c)
        return self.encoder(batch_of_keypoints_cropped)

    def decode(self, z): # P(x|z, c)
        return self.decoder(z)

    def forward(self, batch_of_keypoints_cropped, noise):#noise not used here but for compatibility with others
        return self.decode(self.encode(batch_of_keypoints_cropped))

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
        netG_ = Generator(ngpu, numJoints, nz)
        netG = netG_.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))
        netG.apply(train_utils.weights_init)
        print(netG)

        self.netG = netG
        self.KEYPOINT_RESTORATION = KEYPOINT_RESTORATION



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
        self.optimizerG = optim.Adam(models.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        self.nz = nz


def trainStep(models, trainSetup, b_size, device, tb, step_absolute, num_epochs, epoch, i, batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file):
    
    models.netG.zero_grad()
    
    # Generate batch of latent vectors
    noise = torch.randn(b_size, trainSetup.nz, device=device)
    
    # Generate fake image batch with G
    batch_of_fake_original = models.netG(batch_of_keypoints_cropped, noise)

    g_pixel = trainSetup.lossFunctionG_regression(batch_of_fake_original, batch_of_keypoints_original) #pixel loss

    errG = g_pixel

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
