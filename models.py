from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb

##############################
#        Encoder 
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        """ The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
            This encoder uses resnet-18 to extract features, and further encode them into a distribution
            similar to VAE encoder. 

            Note: You may either add "reparametrization trick" and "KL divergence" or in the train.py file
            
            Args in constructor: 
                latent_dim: latent dimension for z 
  
            Args in forward function: 
                img: image input (from domain B)

            Returns: 
                mu: mean of the latent code 
                logvar: sigma of the latent code 
        """

        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)      
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar


##############################
#        Generator 
##############################

def crop_image(tensor, target_tensor):
    #converting shape of tensor to target tensor shape   
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta] 

def double_convolution(inchannels, outchannels, normalize=True):
    #in downconv, we will not normalize in top and final layers of it
    #in upconv, we always use batchnorm
    if normalize:
        return nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=(3, 3), padding=1),
            nn.InstanceNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannels, out_channels=outchannels, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )    



class Generator(nn.Module):
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B
        
        Args in constructor: 
            latent_dim: latent dimension for z 
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
        
        Args in forward function: 
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns: 
            fake_B: generated image in domain B
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape
        
        # Define UNET architecture
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride = 2)
        self.fc = nn.Linear(latent_dim,self.h * self.w) #converts z from 1d to 2d

        #downsampling part
        self.downconv1 = double_convolution(channels + 1,64,normalize=False) #channels +1 because extra depth of z is also added
        self.downconv2 = double_convolution(64,128,normalize=True)
        self.downconv3 = double_convolution(128,256,normalize=True)
        self.downconv4 = double_convolution(256,512,normalize=True)
        self.downconv5 = double_convolution(512,1024,normalize=False)
        

        #upsampling part
        #in channels below -> we also incorporate z feature
        self.transposeconv1 = nn.ConvTranspose2d(in_channels=1024,out_channels = 512,kernel_size=(2,2),stride = 2)
        #double convolutions again
        self.upconv1 = double_convolution(1024,512,normalize=True) #1024 because after skip connections -> 512+512
        self.transposeconv2 = nn.ConvTranspose2d(in_channels=512,out_channels = 256,kernel_size=(2,2),stride = 2)
        self.upconv2 = double_convolution(512,256,normalize=True)
        self.transposeconv3 = nn.ConvTranspose2d(in_channels=256,out_channels = 128,kernel_size=(2,2),stride = 2)
        self.upconv3 = double_convolution(256,128,normalize=True)
        self.transposeconv4 = nn.ConvTranspose2d(in_channels=128,out_channels = 64,kernel_size=(2,2),stride = 2)
        self.upconv4 = double_convolution(128,64,normalize=True)
        
        #1*1 convolution
        self.final_conv = nn.Conv2d(in_channels = 64, out_channels = channels,kernel_size = 1)

    def forward(self, x, z):
        print("x shape initially",x.shape)  #torch.Size([4, 3, 256, 256])
        print("z shape initially",z.shape)  #torch.Size([4, 8])
        
        # convert 2d z to 4d z so that it can be concatenated with image
        z = self.fc(z).view(z.size(0), 1, self.h, self.w)
        print(z.shape)                      #torch.Size([4, 1, 256, 256])
        
        #now this z is concatenated with x
        x1 = self.downconv1(torch.cat([x, z], dim=1))
        print(x1.shape)                     #torch.Size([4, 64, 256, 256])
        x2 = self.maxpool(x1)
        print(x2.shape)                     #torch.Size([4, 64, 128, 128])
        x3 = self.downconv2(x2)
        print(x3.shape)                     #torch.Size([4, 128, 128, 128])
        x4 = self.maxpool(x3)
        print(x4.shape)                     #torch.Size([4, 128, 64, 64])
        x5 = self.downconv3(x4)
        print(x5.shape)                     #torch.Size([4, 256, 64, 64])
        x6 = self.maxpool(x5)
        print(x6.shape)                     #torch.Size([4, 256, 32, 32])
        x7 = self.downconv4(x6)
        print(x7.shape)                     #torch.Size([4, 512, 32, 32])
        x8 = self.maxpool(x7)
        print(x8.shape)                     #torch.Size([4, 512, 16, 16])
        x9 = self.downconv5(x8)
        print(x9.shape)                     #torch.Size([4, 1024, 16, 16])
        
              
        #upsampling
        x = self.transposeconv1(x9)
        print(x.shape)                      #torch.Size([4, 512, 32, 32])
        #we need to crop x7 to fit x's size before we can perform skip connections concatenation
        y = crop_image(x7,x)
        print(y.shape)                      #torch.Size([4, 512, 32, 32]) 
        x = self.upconv1(torch.cat([x,y],1))  #This step involves concatenating 512 channels + 512 channels but i also do upconv so resultant will have 512 channels again as in unet paper
        print(x.shape)                      #torch.Size([4, 512, 32, 32])
        
        x = self.transposeconv2(x)   
        print(x.shape)                      #torch.Size([4, 256, 64, 64])
        #but before that we need to crop x5 to fit x's size
        y = crop_image(x5,x)
        x = self.upconv2(torch.cat([x,y],1))
        print(x.shape)                      #torch.Size([4, 256, 64, 64])
        
        x = self.transposeconv3(x) 
        print(x.shape)                      #    torch.Size([4, 128, 128, 128])
        #but before that we need to crop x3 to fit x's size
        y = crop_image(x3,x)
        x = self.upconv3(torch.cat([x,y],1))
        print(x.shape)                      #torch.Size([4, 128, 128, 128])
        
        x = self.transposeconv4(x)  
        print(x.shape)                      #torch.Size([4, 64, 256, 256]) 
        #but before that we need to crop x1 to fit x's size
        y = crop_image(x1,x)
        x = self.upconv4(torch.cat([x,y],1))
        print(x.shape)                      #torch.Size([4, 64, 256, 256])
        
        #output
        x = self.final_conv(x)              #torch.Size([4, 3, 256, 256])
        return x


##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
          super(Discriminator, self).__init__()
          """ The discriminator used in both cVAE-GAN and cLR-GAN
          This code implements a 70x70 patchGAN

              Args in constructor:
                  in_channels: number of channel in image (default: 3 for RGB)

              Args in forward function:
                  x: image input (real_B, fake_B)

              Returns:
                  discriminator output: could be a single value or a matrix depending on the type of GAN
          """
          # Convolutional layers
          self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
          self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
          self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
          self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)

          # Final convolutional layer for patch classification
          self.patch_classifier = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

          # Activation function (Leaky ReLU)
          self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

          # Batch normalization
          self.batch_norm2d_1 = nn.BatchNorm2d(128)
          self.batch_norm2d_2 = nn.BatchNorm2d(256)

          # Number of patches to classify

    def forward(self, x):
          x = self.leaky_relu(self.conv1(x))
          x = self.leaky_relu(self.batch_norm2d_1(self.conv2(x)))
          x = self.leaky_relu(self.batch_norm2d_2(self.conv3(x)))
          x = self.leaky_relu(self.conv4(x))
          
          # Patch classification
          patch_scores = self.patch_classifier(x)

          return patch_scores
if __name__ == "__main__":

    def test_discriminator():
        D = Discriminator()
        im_test = torch.randn((4,3,256,256))
        res = D(im_test)
        print(res.shape)
    
    








