import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from torch import nn, optim
from vis_tools import *
from datasets import *
from models import *
import argparse, os
import itertools
import torch
import time
import pdb

# Training Configurations
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
img_dir = '/home/zlz/BicycleGAN/datasets/edges2shoes/train/'
img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
num_epochs = 2
batch_size = 1
lr_rate = 0.0002  	      # Adam optimizer learning rate
betas = (0.5, 0.999)			  # Adam optimizer beta 1, beta 2
lambda_pixel =  10      # Loss weights for pixel loss
lambda_latent = 0.5    # Loss weights for latent regression
lambda_kl =  0.01      # Loss weights for kl divergence
latent_dim =   8       # latent dimension for the encoded images from domain B
gpu_id = '0'

#NOTE: Currently using only single learning rate, can change this

# Normalize image tensor
def norm(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
	return ((tensor+1.0)/2.0)*255.0

# Reparameterization helper function
# (You may need this helper function here or inside models.py, depending on your encoder implementation)


# Random seeds (optional)
torch.manual_seed(1); np.random.seed(1)

# Define DataLoader
dataset = Edge2Shoe(img_dir)
loader = data.DataLoader(dataset, batch_size=batch_size)

# Loss functions
mae_loss = torch.nn.L1Loss().to(gpu_id)
mse_loss = torch.nn.MSELoss().to(gpu_id)

# Define generator, encoder and discriminators
generator = Generator(latent_dim, img_shape).to(gpu_id)
encoder = Encoder(latent_dim).to(gpu_id)
D_VAE = Discriminator().to(gpu_id)
D_LR = Discriminator().to(gpu_id)

# Define optimizers for networks
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=betas)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=betas)
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate, betas=betas)
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=lr_rate, betas=betas)

# For adversarial loss (optional to use)
valid = 1; fake = 0

# Training
total_steps = len(loader)*num_epochs; step = 0
for e in range(num_epochs):
	start = time.time()
	for idx, data in enumerate(loader):
		

		#NOTE: Do a zero grad here somewhere
		########## Process Inputs ##########
		edge_tensor, rgb_tensor = data
		edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
		real_A = edge_tensor; real_B = rgb_tensor;

		#Encoder & Generator Loss
  
		#first we pass real B to encoder in cvae-gan 
		z_encoded,mu,logvar = encoder(real_B)
		# next we pass sketch image, z to generator to get fakeB
		fake_B_vae = generator(real_A, z_encoded)
		#next, we calculate L1 loss
		cvaegan_l1  = mae_loss(fake_B_vae,real_B)
		#next we calculate kl div loss
		KLD = (1/2) * (torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar)  )
  
		#we calculate l1 loss for clrgan
		z_random = torch.randn_like(z_encoded)
		fake_B_clr = generator(real_A,z_random)
		mu_clr, logvar_clr = encoder(fake_B_clr)
		clrgan_l1 = mae_loss(mu_clr, z_random)
		#back propagate

		loss_G = lambda_pixel*cvaegan_l1 + lambda_latent*clrgan_l1 + lambda_kl*KLD
		loss_G.backward()
		optimizer_G.step()
		optimizer_E.step() 

		z_encoded,mu,logvar =  encoder(real_B)
		fake_B_vae = generator(real_A,z_encoded)
		z_random = torch.randn_like(z_encoded)
		fake_B_clr = generator(real_A,z_random)
	
		
	
		#NOTE: Using same data for clrGAN and VAEGAN here. This is against recommendation of eveningglow
		
		#-------------------------------
		#  Train Generator and Encoder
		#------------------------------

		




		#----------------------------------
		#  Train Discriminator (cVAE-GAN)
		#----------------------------------
		
		
		real_D_VAE_scores = D_VAE(real_B)
		fake_D_VAE_scores = D_VAE(fake_B_vae)
	


		D_loss_cVAE = mse_loss(real_D_VAE_scores,torch.ones_like(real_D_VAE_scores)) + mse_loss(fake_D_VAE_scores,torch.zeros_like(real_D_VAE_scores))

		real_D_LR_scores = D_LR(real_B)
		fake_D_LR_scores = D_LR(fake_B_clr)

		D_loss_cLR = mse_loss(real_D_LR_scores,torch.ones_like(real_D_LR_scores)) + mse_loss(fake_D_LR_scores,torch.zeros_like(real_D_LR_scores))

		D_loss = D_loss_cVAE + D_loss_cLR

		D_loss.backward()

		optimizer_D_LR.step()
		optimizer_D_VAE.step()







		#---------------------------------
		#  Train Discriminator (cLR-GAN)
		#---------------------------------





		""" Optional TODO:
			1. You may want to visualize results during training for debugging purpose
			2. Save your model every few iterations
		"""
if __name__ == "__main__":
	pass