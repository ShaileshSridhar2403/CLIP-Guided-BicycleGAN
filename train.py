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
from tqdm import tqdm
import pdb

# Training Configurations
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
img_dir = 'C:/Users/yasha/Desktop/sem3/cis680/shailesh/CLIP-Guided-BicycleGAN/edges2shoes/edges2shoes/train/'
img_shape = (3, 128, 128) # Please use this image dimension faster training purpose
num_epochs = 2
batch_size = 1
lr_rate = 0.0002  	      # Adam optimizer learning rate
betas = (0.5, 0.999)			  # Adam optimizer beta 1, beta 2
lambda_pixel =  10      # Loss weights for pixel loss
lambda_latent = 0.5    # Loss weights for latent regression
lambda_kl =  0.01      # Loss weights for kl divergence
latent_dim =   8       # latent dimension for the encoded images from domain B
gpu_id = 'cuda:0'

# Normalize image tensor
def norm(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
	return ((tensor+1.0)/2.0)*255.0

# Reparameterization helper function
# (You may need this helper function here or inside models.py, depending on your encoder implementation)
#implemented in encoder

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

#initialize losses
running_total_loss = 0
running_loss_cvaegan_l1 = 0
running_loss_KLD = 0
running_loss_D_loss_cVAE = 0
running_loss_D_loss_cLRGAN = 0
running_loss_clrgan_l1 = 0

total_loss_history = []
loss_cvaegan_l1_history = []
loss_KLD_history = []
loss_D_loss_cVAE_history = []
loss_D_loss_cLRGAN_history = []
loss_clrgan_l1_history = []




# Training
def train(argpath = None):
	if argpath is not None:
		img_dir = argpath
	total_steps = len(loader)*num_epochs; step = 0
	for e in range(num_epochs):
		start = time.time()
		tqdm_loader = tqdm(loader, total=len(loader))
		for idx, data in enumerate(tqdm_loader):  
			edge_tensor, rgb_tensor = data
			edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
			real_A = edge_tensor; real_B = rgb_tensor;

			#-------------------------------
			#  Train Generator and Encoder
			#------------------------------

			#Encoder & Generator Loss

			#zero grad
			optimizer_G.zero_grad()
			optimizer_E.zero_grad()
			optimizer_D_VAE.zero_grad()
			optimizer_D_LR.zero_grad()
            

			#first we pass real B to encoder in cvae-gan
			z_encoded,mu,logvar = encoder(real_B)
			# next we pass sketch image, z to generator to get fakeB
			fake_B_vae = generator(real_A, z_encoded)
	
	
			#we calculate l1 loss for clrgan
			z_random = torch.randn_like(z_encoded)
			fake_B_clr = generator(real_A,z_random)
			_, mu_clr, logvar_clr2 = encoder(fake_B_clr)
			clrgan_l1 = mae_loss(mu_clr, z_random)
			clrgan_l1.backward()
			optimizer_G.step()

	
			#next, we calculate cvaegan L1 loss
			cvaegan_l1  = mae_loss(fake_B_vae,real_B)  #got cvaegan L1 loss
			#next we calculate kl div loss
			KLD = (1/2) * (torch.sum(torch.exp(logvar) + mu.pow(2) - 1 - logvar)  )  #got cvaegan KL loss

			#calculate cvae gan L2 loss
			real_D_VAE_scores = D_VAE(real_B)
			fake_D_VAE_scores = D_VAE(fake_B_vae)
			real_D_LR_scores = D_LR(real_B)
			fake_D_LR_scores = D_LR(fake_B_clr)

			VAE_GAN_loss =  mse_loss(fake_D_VAE_scores,torch.ones_like(fake_D_VAE_scores))
			LR_GAN_loss = mse_loss(fake_D_LR_scores, torch.ones_like(fake_D_VAE_scores) )

			loss_GE = VAE_GAN_loss + LR_GAN_loss + cvaegan_l1*lambda_pixel + KLD * lambda_kl

			loss_GE.backward()
			optimizer_E.step()


			

			#optimizing discriminators
			

			D_loss_cVAE = mse_loss(real_D_VAE_scores,torch.ones_like(real_D_VAE_scores)) + mse_loss(fake_D_VAE_scores,torch.zeros_like(fake_D_VAE_scores))
			D_loss_cVAE.backward()
			optimizer_D_VAE.step()


			D_loss_cLRGAN = mse_loss(real_D_LR_scores,torch.ones_like(real_D_LR_scores)) + mse_loss(fake_D_LR_scores,torch.zeros_like(fake_D_LR_scores))
			D_loss_cLRGAN.backward()
			optimizer_D_LR.step()


			"""
			#add all losses
			running_total_loss += (D_loss_cVAE +D_loss_cVAE + cvaegan_l1*lambda_pixel + KLD * lambda_kl + clrgan_l1*lambda_latent).item()
			running_loss_cvaegan_l1 += cvaegan_l1.item()
			running_loss_KLD += KLD.item()
			running_loss_D_loss_cVAE += D_loss_cVAE.item()
			running_loss_D_loss_cLRGAN += D_loss_cLRGAN.item()
			running_loss_clrgan_l1 = clrgan_l1.item()


			########## Visualization ##########
			if step % report_feq == report_feq-1:
				print('Train Epoch: {} {:.0f}% \tTotal Loss: {:.6f} \loss_cvaegan_l1: {:.6f}\loss_KLD: {:.6f}\D_loss_cVAE: {:.6f}\D_loss_cLRGAN: {:.6f}\loss_clrgan_l1: {:.6f}'.format
						(e+1, 100. * idx / len(loader), running_total_loss / report_feq,
						running_loss_cvaegan_l1/report_feq, running_loss_KLD/report_feq,
						running_loss_D_loss_cVAE/report_feq, running_loss_D_loss_cLRGAN/report_feq,
						running_loss_clrgan_l1/report_feq))

				#now store losses in list for plot
				total_loss_history.append(running_total_loss/report_feq)
				loss_cvaegan_l1_history.append(running_loss_cvaegan_l1/report_feq)
				loss_KLD_history.append(running_loss_KLD/report_feq)
				loss_D_loss_cVAE_history.append(running_loss_D_loss_cVAE/report_feq)
				loss_D_loss_cLRGAN_history.append(running_loss_D_loss_cLRGAN/report_feq)
				loss_clrgan_l1_history.append(running_loss_clrgan_l1/report_feq)

				#now reset once saved
				running_loss_D_A = 0
				running_loss_D_B = 0
				running_loss_GAN_AB = 0
				running_loss_GAN_BA = 0
				running_loss_cycle = 0
				running_total_loss = 0
				end = time.time()
				print(e, step, 'T: ', end-start)
				start = end

				#Visualize generated images

				"""





















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

if __name__ == '_main_':
    train()