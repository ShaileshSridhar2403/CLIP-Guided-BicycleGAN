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
    
    def reparameterization(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = self.reparameterization(mu,logvar)

        return z


##############################
#        Generator 
##############################
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
        # (TODO: add layers...)

    def forward(self, x, z):
        # (TODO: add layers...)

        return 

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
    









