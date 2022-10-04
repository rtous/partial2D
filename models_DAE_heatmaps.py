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
import colors

#pix2pix: https://github.com/znxlwm/pytorch-pix2pix/blob/master/network.py


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

NEURONS_PER_LAYER_GENERATOR = 4#32

def noiseSquareBatch(outputRes, batchsize):
    #noiseSquareBatch = np.zeros((batchsize, 1, outputRes, outputRes), dtype="float32")
    noiseSquareBatch = np.random.random_sample((batchsize, 1, outputRes, outputRes)).astype('float32')
    return noiseSquareBatch

class Generator(nn.Module):
    #Channels = nc + 1 (1 for the noise)
    def __init__(self, channels, addNoise=False):
        super(Generator, self).__init__()
        self.channels = channels
        self.addNoise = addNoise
        if self.addNoise:
            self.input_channels = self.channels+1
        else:
            self.input_channels = self.channels  

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
            *downsample(self.input_channels, 32, normalize=True),
            *downsample(32, 64),
            *downsample(64, 1000, normalize=True, relu=True),
            #nn.Conv2d(128, 1000, kernel_size=4, stride=2, padding=1),
            *upsample(1000, 64),
            *upsample(64, 32),
            *upsample(32, self.channels, normalize=False, relu=False),
            #nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() 
            #nn.Sigmoid()
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
    #About noise in cGAN: https://arxiv.org/pdf/1905.02135.pdf
    #For instance, assume the shape of current layer is C × H × W
    #the Gaussian noise nz × 1 × 1 is spatially replicated to nz × H × W along H and W dimension. 
    #After concatenation along channel dimension, the shape of resulting concatenated layer will be (C + nz) × H × W.
    def forward(self, input):
        print("Generator received: ", input.shape)
        #print("Generator...")
        batchsize = input.shape[0]
        channels = input.shape[1]
        outputRes = input.shape[2]
        print("batchsize=", batchsize)
        print("outputRes=", outputRes)
        if self.addNoise:
            #noise = torch.randn(batchsize, channels, 1, 1)
            noise = torch.FloatTensor(batchsize, 1, outputRes, outputRes).normal_(0, 1)
            #noise = torch.Variable(noise)
            #noise = torch.tensor(noiseSquareBatch(outputRes, batchsize))
            input = torch.cat((input, noise), 1) 
        else:
            print("WARING: Not adding noise to the input.") 
        return self.model(input)


NEURONS_PER_LAYER_DISCRIMINATOR = 4#16
class Discriminator64(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator64, self).__init__()
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

#NEURONS_PER_LAYER_DISCRIMINATOR = 4
class Discriminator128(nn.Module):
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
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(in_channels=ndf * 4, out_channels=1, kernel_size=(2,2), stride=2, padding=0, bias=False),
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
    batch_of_fake_original_ = batch_of_fake_original.clone()

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

class Models():
    #Receives a noise vector (nz dims) + keypoints cropped (50 dims)
    def __init__(self, ngpu, numJoints, nz, KEYPOINT_RESTORATION, device = torch.device("cpu")):
        super(Models, self).__init__()
        #generator
        #netG_ = Generator(ngpu, numJoints, nz)
        netG_ = Generator(channels=numJoints)
        netG = netG_.to(device)
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))
        netG.apply(train_utils.weights_init)
        print(netG)

        #discriminator
        netD_ = Discriminator64(ngpu)
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

        print(colors.CRED + "run export OMP_NUM_THREADS=1 in the terminal to avoid parallelization warning and block" + colors.CEND)

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
        self.optimizerD = optim.Adam(models.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(models.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        self.nz = nz
        self.PIXELLOSS_WEIGHT = PIXELLOSS_WEIGHT


def trainStep(models, trainSetup, b_size, device, tb, step_absolute, num_epochs, epoch, i, batch_of_keypoints_cropped, batch_of_keypoints_original, confidence_values, scaleFactor, x_displacement, y_displacement, batch_of_json_file):
    ## Train with all-real batch
    models.netD.zero_grad()
 
    #Batch of real labels
    label = torch.full((b_size,), trainSetup.real_label, dtype=torch.float, device=device)
    
    # Forward pass real batch through D
    output = models.netD(batch_of_keypoints_cropped, batch_of_keypoints_original).view(-1)
    
    # Calculate loss on all-real batch
    errD_real = trainSetup.lossFunctionD(output, label)

    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    noise = torch.randn(b_size, trainSetup.nz, device=device)
    
    # Generate fake image batch with G
    batch_of_fake_original = models.netG(batch_of_keypoints_cropped)

    #Restore the original keypoints with confidence > CONFIDENCE_THRESHOLD_TO_KEEP_JOINTS
    if models.KEYPOINT_RESTORATION:
        batch_of_fake_original = restoreOriginalKeypoints(batch_of_fake_original, batch_of_keypoints_cropped, confidence_values)

    #As they are fake images let's prepare a batch of labels FAKE
    label.fill_(trainSetup.fake_label)
   
    # Classify all fake batch with D
    output = models.netD(batch_of_keypoints_cropped, batch_of_fake_original.detach()).view(-1)
    
    # Calculate D's loss on the all-fake batch
    errD_fake = trainSetup.lossFunctionD(output, label)
    
    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    errD_fake.backward()
    
    D_G_z1 = output.mean().item()
    
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    
    # Update D
    trainSetup.optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    models.netG.zero_grad()
    
    label.fill_(trainSetup.real_label)  # fake labels are real for generator cost
    
    # Since we just updated D, perform another forward pass of all-fake batch through D
    output = models.netD(batch_of_keypoints_cropped, batch_of_fake_original).view(-1)
    
    # Calculate G's loss based on this output
    #errG = criterion(output, label)

    ##############

    g_adv = trainSetup.lossFunctionG_adversarial(output, label) #adversarial loss
    g_pixel = trainSetup.lossFunctionG_regression(batch_of_fake_original, batch_of_keypoints_original) #pixel loss

    errG = (1-trainSetup.PIXELLOSS_WEIGHT) * g_adv + trainSetup.PIXELLOSS_WEIGHT * g_pixel

    
    ###############

    # Calculate gradients for G
    errG.backward()
    
    D_G_z2 = output.mean().item()
    
    # Update G
    trainSetup.optimizerG.step()

    tb.add_scalar("LossG", errG.item(), step_absolute)
    tb.add_scalar("g_adv", g_adv.item(), step_absolute)
    tb.add_scalar("g_pixel", g_pixel.item(), step_absolute)
    tb.add_scalar("LossD", errD.item(), step_absolute)
    tb.add_scalar("errD_real", errD_real.item(), step_absolute)
    tb.add_scalar("errD_fake", errD_fake.item(), step_absolute)

    # Output training stats each 50 batches
        
    if i % 50 == 0:
        print("**************************************************************")
        print('[%d/%d][%d/?]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, #len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        print('errD_real: %.4f, errD_fake: %.4f\t'
              % (errD_real.item(), errD_fake.item()))

        print('loss g_adv: %.4f, loss g_pixel: %.4f\t'
              % (g_adv.item(), g_pixel.item()))

def inference(models, b_size, fixed_noise, numJoints, batch_of_keypoints_cropped, confidence_values):
    with torch.no_grad():
        print("drawing batch...")
        fake = models.netG(batch_of_keypoints_cropped).detach().cpu()
        
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