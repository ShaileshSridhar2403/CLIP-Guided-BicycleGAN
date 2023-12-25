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

        return z,mu,logvar


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



# This is similar to Cyclegan generator except we incorporate latent dimension
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, input_shape, num_residual_blocks, device='cuda'):
        super(Generator, self).__init__()

        self.device = device

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels + latent_dim, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            #nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
            z.size(0), z.size(1), x.size(2), x.size(3))
        out = torch.cat([x, z_img], dim=1)
        return self.model(out)


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

    #code to test generator
    # Testing generator
    batch_size = 4
    channels = 3
    height = 256
    width = 256
    latent_dim = 8
    mock_images = torch.randn(batch_size, channels, height, width)
    mock_latent = torch.randn(batch_size, latent_dim)

    model = Generator(latent_dim=latent_dim, img_shape=(channels, height, width))
    with torch.no_grad():
        output = model(mock_images, mock_latent)
        print("Output Shape:", output.shape)

    









