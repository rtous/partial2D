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


NEURONS_PER_LAYER_GENERATOR = 512
class Generator(nn.Module):
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, ngpu, numJoints, nz):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.image_size = numJoints*2
        self.main = nn.Sequential(
          # First upsampling
          nn.Linear(self.image_size+nz, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Second upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Third upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, NEURONS_PER_LAYER_GENERATOR, bias=False),
          nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          nn.LeakyReLU(0.25),
          # Third upsampling
          #nn.Linear(NEURONS_PER_LAYER_GENERATOR, NEURONS_PER_LAYER_GENERATOR, bias=False),
          #nn.BatchNorm1d(NEURONS_PER_LAYER_GENERATOR, 0.8),
          #nn.LeakyReLU(0.25),
          # Final upsampling
          nn.Linear(NEURONS_PER_LAYER_GENERATOR, self.image_size, bias=False),
          #nn.Tanh()
        )

    def forward(self, batch_of_keypoints_cropped, noise):
        input = torch.cat((batch_of_keypoints_cropped, noise), -1)
        return self.main(input)

NEURONS_PER_LAYER_DISCRIMINATOR = 512
DISCRIMINATOR_OUTPUT_SIZE = 1
class Discriminator(nn.Module):
    def __init__(self, ngpu, numJoints):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.image_size = numJoints*2
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(16,16), stride=2, padding=1, bias=False),
            nn.Linear(self.image_size*2, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            #nn.BatchNorm1d(NEURONS_PER_LAYER_DISCRIMINATOR, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.LeakyReLU(0.2),

            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            #nn.BatchNorm1d(DISCRIMINATOR_OUTPUT_SIZE, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            #nn.BatchNorm1d(DISCRIMINATOR_OUTPUT_SIZE, 0.8),
            #nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            #nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, DISCRIMINATOR_OUTPUT_SIZE, bias=False),
            #nn.BatchNorm1d(DISCRIMINATOR_OUTPUT_SIZE, 0.8),
            #nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, batch_of_keypoints_cropped, batch_of_keypoints_original): 
        #print("Discriminator batch_of_keypoints_cropped shape:",batch_of_keypoints_cropped.shape)
        #print("Discriminator batch_of_keypoints_original shape:",batch_of_keypoints_original.shape)
        input = torch.cat((batch_of_keypoints_cropped, batch_of_keypoints_original), -1)  
        #print("Discriminator input:",input.shape)
        #print("D input: ", input[0])
        return self.main(input)
'''
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=(16,16), stride=2, padding=1, bias=False),
            nn.Linear(image_size*2, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            nn.BatchNorm1d(NEURONS_PER_LAYER_DISCRIMINATOR, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            #nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            nn.BatchNorm1d(NEURONS_PER_LAYER_DISCRIMINATOR, 0.8),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, NEURONS_PER_LAYER_DISCRIMINATOR, bias=False),
            nn.BatchNorm1d(NEURONS_PER_LAYER_DISCRIMINATOR, 0.8),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=2, stride=1, padding=0, bias=False),
            nn.Linear(NEURONS_PER_LAYER_DISCRIMINATOR, 1, bias=False),
            #nn.Sigmoid() 
            #idea: supress sigmoid from https://github.com/soumith/ganhacks/issues/36
        )

    def forward(self, batch_of_keypoints_cropped, batch_of_keypoints_original): 
        #print("Discriminator batch_of_keypoints_cropped shape:",batch_of_keypoints_cropped.shape)
        #print("Discriminator batch_of_keypoints_original shape:",batch_of_keypoints_original.shape)
        input = torch.cat((batch_of_keypoints_cropped, batch_of_keypoints_original), -1)  
        #print("Discriminator input:",input.shape)
        print("D input: ", input[0])
        return self.main(input)
'''
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

        #discriminator
        netD_ = Discriminator(ngpu, numJoints)
        netD = netD_.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))
        netD.apply(train_utils.weights_init)
        print(netD)
        #pytorchUtils.explainModel(netD, 28, 28, 1, 1)
        #if arguments.debug:
        #    pytorchUtils.registerDebugHook(netD_)

        self.netG = netG
        self.netD = netD
        self.KEYPOINT_RESTORATION = KEYPOINT_RESTORATION


class TrainSetup():
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, models, ngpu, numJoints, nz, lr, beta1, PIXELLOSS_WEIGHT, device = "cpu"):
        super(TrainSetup, self).__init__()

        self.optimizerG = torch.optim.RMSprop(models.netG.parameters(), lr=lr)
        self.optimizerD = torch.optim.RMSprop(models.netD.parameters(), lr=lr)

        self.nz = nz
        
def trainStep(models, trainSetup, b_size, device, tb, step_absolute, num_epochs, epoch, i, batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file):
    
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    models.netD.zero_grad()

    # Generate fake image batch with G
    # Generate batch of latent vectors
    noise = torch.randn(b_size, trainSetup.nz, device=device)
    batch_of_fake_original = models.netG(batch_of_keypoints_cropped, noise)
    #print("batch_of_fake_original[0]:", batch_of_fake_original[0])
    #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
    #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
    if models.KEYPOINT_RESTORATION:
        batch_of_fake_original = restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, confidence_values)
    
    #print("batch_of_fake_original[0] restored:", batch_of_fake_original[0])

    # Forward pass real batch through D
    D_real = models.netD(batch_of_keypoints_cropped, batch_of_keypoints_original).view(-1)
    # Classify all fake batch with D
    D_fake = models.netD(batch_of_keypoints_cropped, batch_of_fake_original.detach()).view(-1)
    
    #print("D_real", D_real.detach())
    #print("D_fake", D_fake.detach())

     # Compute error of D as sum over the fake and the real batches
    errD = -(torch.mean(D_real) - torch.mean(D_fake))
    errD.backward()
    trainSetup.optimizerD.step()

    #Do this also here to plot errG all iterations (it's only computed in some)
    errG = -torch.mean(models.netD(batch_of_keypoints_cropped, batch_of_fake_original))           
    
    if i % 3 == 0:
        print("training generator")
        models.netG.zero_grad()
        noise = torch.randn(b_size, trainSetup.nz, device=device)
        batch_of_fake_original = models.netG(batch_of_keypoints_cropped, noise)
        #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
        if models.KEYPOINT_RESTORATION:
            batch_of_fake_original = restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, confidence_values)
    
        errG = -torch.mean(models.netD(batch_of_keypoints_cropped, batch_of_fake_original))           
        errG.backward()
        trainSetup.optimizerG.step()

    tb.add_scalar("LossG", errG.item(), step_absolute)
    #tb.add_scalar("g_adv", g_adv.item(), step_absolute)
    #tb.add_scalar("g_pixel", g_pixel.item(), step_absolute)
    tb.add_scalar("LossD", errD.item(), step_absolute)
    #tb.add_scalar("errD_real", errD_real.item(), step_absolute)
    #tb.add_scalar("errD_fake", errD_fake.item(), step_absolute)

    # Output training stats each 50 batches
    if i % 50 == 0:
        print("**************************************************************")
        print('[%d/%d][%d/?]\tLoss_D: %.4f\tLoss_G: %.4f\t'
              % (epoch, num_epochs, i, #len(dataloader),
                 errD.item(), errG.item()))            

    
def inference(models, b_size, fixed_noise, numJoints, batch_of_keypoints_cropped, confidence_values):
    #receives normalized and returns a normalized value (needs to be normalized)
    with torch.no_grad():
        print("drawing batch...")
        fake = models.netG(batch_of_keypoints_cropped, fixed_noise).detach().cpu()
        #We restore the original keypoints (before denormalizing)
        if models.KEYPOINT_RESTORATION:
            fake = restoreOriginalKeypoints(fake, batch_of_keypoints_cropped, confidence_values)
        print("Shape of fake: ", fake.shape)
        

        #fakeReshapedAsKeypoints = np.reshape(fake, (b_size, numJoints, 2))
        #fakeReshapedAsKeypoints = fakeReshapedAsKeypoints.numpy()
        fakeReshapedAsKeypoints = fake
        
        return fakeReshapedAsKeypoints

def save(models, OUTPUTPATH, epoch, i):
    torch.save(models.netG.state_dict(), OUTPUTPATH+"/model/model_epoch"+str(epoch)+"_batch"+str(i)+".pt")

def load(models, MODELPATH):
    models.netG.load_state_dict(torch.load(MODELPATH))
