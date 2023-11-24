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
num_epochs =  
batch_size = 
lr_rate =   	      # Adam optimizer learning rate
betas = 			  # Adam optimizer beta 1, beta 2
lambda_pixel =        # Loss weights for pixel loss
lambda_latent =       # Loss weights for latent regression 
lambda_kl =           # Loss weights for kl divergence
latent_dim =          # latent dimension for the encoded images from domain B
gpu_id = 

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

		########## Process Inputs ##########
		edge_tensor, rgb_tensor = data
		edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
		real_A = edge_tensor; real_B = rgb_tensor;

		#-------------------------------
		#  Train Generator and Encoder
		#------------------------------


		


		#----------------------------------
		#  Train Discriminator (cVAE-GAN)
		#----------------------------------


		


		#---------------------------------
		#  Train Discriminator (cLR-GAN)
		#---------------------------------


		


		""" Optional TODO: 
			1. You may want to visualize results during training for debugging purpose
			2. Save your model every few iterations
		"""
		






