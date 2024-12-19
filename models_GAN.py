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

'''
NEURONS_PER_LAYER_GENERATOR = 32 #256
BIAS = False
class Generator(nn.Module):
    def __init__(self, ngpu, numJoints, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.numJoints = numJoints
        self.image_size = self.numJoints*2
        self.nz = nz
        self.encoder = nn.Sequential(
          
          nn.Linear(self.image_size, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, int(NEURONS_PER_LAYER_GENERATOR*2), bias=False),
          nn.BatchNorm1d(int(NEURONS_PER_LAYER_GENERATOR*2), 0.8),
          nn.LeakyReLU(0.25),

          nn.Linear(int(NEURONS_PER_LAYER_GENERATOR*2), nz, bias=False),
          nn.BatchNorm1d(nz, 0.8),
          nn.LeakyReLU(0.25),
        )
        self.decoder = nn.Sequential(
          
          nn.Linear(nz, int(NEURONS_PER_LAYER_GENERATOR*2), bias=False),
          nn.BatchNorm1d(int(NEURONS_PER_LAYER_GENERATOR*2), 0.8),
          nn.LeakyReLU(0.25),

          nn.Linear(int(NEURONS_PER_LAYER_GENERATOR*2), NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, self.image_size, bias=False),
        )

    def encode(self, batch_of_keypoints_cropped): # Q(z|x, c)
        return self.encoder(batch_of_keypoints_cropped)

    def decode(self, z): # P(x|z, c)
        return self.decoder(z)

    def forward(self, batch_of_keypoints_cropped, noise):#noise not used here but for compatibility with others
        return self.decode(self.encode(batch_of_keypoints_cropped))
'''

NEURONS_PER_LAYER_GENERATOR = 32 #256
BIAS = False
class Generator(nn.Module):
    def __init__(self, ngpu, numJoints, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.numJoints = numJoints
        self.image_size = self.numJoints*2
        self.nz = nz
        self.main = nn.Sequential(
          nn.Linear(nz, int(NEURONS_PER_LAYER_GENERATOR), bias=False),
          nn.BatchNorm1d(int(NEURONS_PER_LAYER_GENERATOR), 0.8),
          #nn.LeakyReLU(0.25),

          nn.Linear(nz, int(NEURONS_PER_LAYER_GENERATOR), bias=False),
          #nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          #nn.LeakyReLU(0.25),
          
          nn.Dropout(0.5),

          nn.Linear(int(NEURONS_PER_LAYER_GENERATOR), NEURONS_PER_LAYER_GENERATOR, bias=False),
          #nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          #nn.LeakyReLU(0.25),
          
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, self.image_size, bias=False),
        )

    def forward(self, noise):
        return self.main(noise)

BIAS = False
class Discriminator(nn.Module):
  def __init__(self, ngpu, numJoints, nz):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.numJoints = numJoints
    self.image_size = self.numJoints*2
    self.nz = nz
    self.main = nn.Sequential(
      nn.Linear(self.image_size, NEURONS_PER_LAYER_GENERATOR, bias=BIAS),
        #nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
        #nn.LeakyReLU(0.2, inplace=True),
        
        nn.Linear(NEURONS_PER_LAYER_GENERATOR, int(NEURONS_PER_LAYER_GENERATOR), bias=BIAS),
        #nn.BatchNorm1d(int(NEURONS_PER_LAYER_GENERATOR), 0.8),
        #nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(NEURONS_PER_LAYER_GENERATOR, int(NEURONS_PER_LAYER_GENERATOR), bias=BIAS),
        #nn.BatchNorm1d(int(NEURONS_PER_LAYER_GENERATOR), 0.8),
        #nn.LeakyReLU(0.2, inplace=True),

        nn.Linear(int(NEURONS_PER_LAYER_GENERATOR), 1, bias=BIAS),
        nn.Sigmoid()
    )

  def forward(self, batch_of_keypoints_cropped):#noise not used here but for compatibility with others
      return self.main(batch_of_keypoints_cropped)


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

        netD_ = Discriminator(ngpu, numJoints, nz)
        netD = netD_.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))
        netD.apply(train_utils.weights_init)
        print(netD)
        self.netD = netD

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
        self.optimizerD = optim.Adam(models.netD.parameters(), lr=lr, betas=(beta1, 0.999))

        self.nz = nz


def trainStep(models, trainSetup, b_size, device, tb, step_absolute, num_epochs, epoch, i, batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file):
    
    #### (1) Train D with a real batch ####
    models.netD.zero_grad()
    #Batch of real labels
    label = torch.full((b_size,), trainSetup.real_label, dtype=torch.float, device=device)
    # Forward pass real batch through D
    output = models.netD(batch_of_keypoints_cropped).view(-1)
    # Calculate loss on all-real batch
    errD_real = trainSetup.lossFunctionD(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()

    ##### (2) Train D with a fake batch (generated with G ####
    # Generate batch of latent vectors
    noise = torch.randn(b_size, trainSetup.nz, device=device)
    # Generate fake image batch with G
    batch_of_fake_original = models.netG(noise)
    #As they are fake images let's prepare a batch of labels FAKE
    label.fill_(trainSetup.fake_label)
    # Classify all fake batch with D
    output = models.netD(batch_of_fake_original.detach()).view(-1)
    # Calculate D's loss on the all-fake batch
    errD_fake = trainSetup.lossFunctionD(output, label)
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # Update D
    trainSetup.optimizerD.step()

    ##### (3) Train G ####
    models.netG.zero_grad()
    # the discriminator results will be compared will a set of "real" labels to penalize those detected as "fake"
    label.fill_(trainSetup.real_label)  
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = models.netD(batch_of_fake_original).view(-1)
    # Calculate G's loss based on this output
    g_adv = trainSetup.lossFunctionG_adversarial(output, label) #adversarial loss
    #g_pixel = trainSetup.lossFunctionG_regression(batch_of_fake_original) #pixel loss
    #errG = (1-trainSetup.PIXELLOSS_WEIGHT) * g_adv + trainSetup.PIXELLOSS_WEIGHT * g_pixel
    errG = g_adv
    # Calculate gradients for G
    errG.backward()
    # Update G
    trainSetup.optimizerG.step()

    tb.add_scalar("LossG", errG.item(), step_absolute)
    tb.add_scalar("LossD", errD.item(), step_absolute)

    # Output training stats each 50 batches
        
    if i % 50 == 0:
        print("**************************************************************")
        print('[epoch %d/%d][batch %d]\tLoss_G: %.4f \tLoss_D: %.4f'
              % (epoch, num_epochs, i, errG.item(), errD.item()))

def inference(models, b_size, noise, numJoints, batch_of_keypoints_cropped, confidence_values):
    #returns normalized and flat values
    with torch.no_grad():
        fake = models.netG(noise).detach().cpu()

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
