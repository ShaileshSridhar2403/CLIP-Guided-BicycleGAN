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

gpu_id = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#gpu_id = 'cuda:0'

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

def plot_loss(total_loss_history,loss_cvaegan_l1_history,
             loss_KLD_history,loss_D_loss_cVAE_history,
             loss_D_loss_cLRGAN_history,
             loss_clrgan_l1_history,loss_vae_gan_loss_history,loss_LR_GAN_loss_history):
    for loss_ind in range(len([total_loss_history,loss_cvaegan_l1_history,
             loss_KLD_history,loss_D_loss_cVAE_history,
             loss_D_loss_cLRGAN_history,
             loss_clrgan_l1_history,
             loss_vae_gan_loss_history,
             loss_LR_GAN_loss_history])):

        loss = [total_loss_history,loss_cvaegan_l1_history,
                loss_KLD_history,loss_D_loss_cVAE_history,
                loss_D_loss_cLRGAN_history,
                loss_clrgan_l1_history,loss_vae_gan_loss_history,loss_LR_GAN_loss_history][loss_ind]
        #print(loss)

        key = ["total_loss_history","loss_cvaegan_l1_history",
                "loss_KLD_history","loss_D_loss_cVAE_history",
                "loss_D_loss_cLRGAN_history",
                "loss_clrgan_l1_history","loss_vae_gan_loss_history","loss_LR_GAN_loss_history"]

        plt.plot(loss,label = key[loss_ind])

        plt.legend()
        plt.show()


def plot_sketch_to_multimodal(generator,sketch,z_dim,n=5):
    sketches = sketch.repeat(n,1,1,1)
    z_random = torch.randn(n,z_dim).to(gpu_id)
    #print(z_random)
    out = generator(sketches,z_random)

    fig, axs = plt.subplots(1,6, figsize = (15,90
                                            ))
    axs[0].imshow(sketch.detach().cpu().numpy().transpose(1,2,0))
    axs[0].set_title('Sketch')
    for i in range(1,n+1):
        img = denorm(out[i-1].squeeze()).cpu().data.numpy().astype(np.uint8)
        axs[i].imshow(img.transpose(1,2,0))  #real sketch
        axs[i].set_title(f'Colored image {i}')

    plt.show()


def calculate_D_loss(fake_B, D, real_B):

    # forward real_B images into the discriminator
    real_D_scores = D(real_B)
    # compute loss between Valid_label and discriminator output on real_B images
    mse_loss_real = mse_loss(real_D_scores, torch.ones_like(real_D_scores))

    # Compute loss between Fake_label and discriminator output on fake images
    fake_D_scores = D(fake_B.detach())
    mse_loss_fake = mse_loss(fake_D_scores, torch.zeros_like(fake_D_scores))
    # sum real_B loss and fake loss as the loss_D
    loss_discriminator = mse_loss_real + mse_loss_fake

    return loss_discriminator


# Training
def train(argpath = None, ckt_path = None):
    if argpath is not None:
        img_dir = argpath
        dataset = Edge2Shoe(img_dir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    total_steps = len(loader)*num_epochs; step = 0
    report_feq = 1000
    save_feq = 6229
    start_epoch = 0

    # Define generator, encoder and discriminators object
    generator = Generator(latent_dim, img_shape, n_residual_blocks, device=gpu_id).to(gpu_id)
    encoder = Encoder(latent_dim).to(gpu_id)
    D_VAE = Discriminator().to(gpu_id)
    D_LR = Discriminator().to(gpu_id)

    # Define optimizers for networks
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=betas)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=betas)
    optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate, betas=betas)
    optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=lr_rate, betas=betas)

    #Initialzing empty loss histories. If checkpoint present, they get replaced below
    total_loss_history = []
    loss_cvaegan_l1_history = []
    loss_KLD_history = []
    loss_D_loss_cVAE_history = []
    loss_D_loss_cLRGAN_history = []
    loss_clrgan_l1_history = []
    loss_vae_gan_loss_history = []
    loss_LR_GAN_loss_history = []

    #Load checkpoint if path is given
    if ckt_path:
        if os.path.isfile(ckt_path):
            print(f"Loading checkpoint '{ckt_path}'")
            ckt = torch.load(ckt_path)
            #load losses
            total_loss_history = ckt['loss_total']
            #print("checkpointed total loss history : ",total_loss_history)
            loss_cvaegan_l1_history = ckt['cvaegan_l1']
            loss_KLD_history = ckt['KLD']
            loss_D_loss_cVAE_history = ckt['D_loss_cVAE']
            loss_D_loss_cLRGAN_history =ckt['D_loss_cLRGAN']
            loss_clrgan_l1_history = ckt['clrgan_l1']
            loss_vae_gan_loss_history = ckt['VAE_GAN_loss']
            loss_LR_GAN_loss_history = ckt['LR_GAN_loss']
            #load gen,disc,encoder objects
            generator.load_state_dict(ckt['generator'])
            encoder.load_state_dict(ckt['encoder'])
            D_VAE.load_state_dict(ckt['D_VAE'])
            D_LR.load_state_dict(ckt['D_LR'])
            #load optimizers
            optimizer_G.load_state_dict(ckt['optimizer_G'])
            optimizer_E.load_state_dict(ckt['optimizer_E'])
            optimizer_D_VAE.load_state_dict(ckt['optimizer_D_VAE'])
            optimizer_D_LR.load_state_dict(ckt['optimizer_D_LR'])
            #load epoch
            start_epoch = ckt['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
          print("checkpoint file corrupted")
    else:
      print("No checkpoints detected. Starting from scratch")

    #common code irrespective of checkpoint
    running_total_loss = 0
    running_loss_cvaegan_l1 = 0
    running_loss_KLD = 0
    running_loss_D_loss_cVAE = 0
    running_loss_D_loss_cLRGAN = 0
    running_loss_clrgan_l1 = 0
    running_loss_vae_gan_loss = 0
    running_loss_LR_GAN_loss = 0

    for e in range(start_epoch, num_epochs):
        tqdm_loader = tqdm(loader, total=len(loader))
        for idx, data in enumerate(tqdm_loader):
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
            real_A = edge_tensor
            real_B = rgb_tensor

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()



            # Encode real images to get latent codes (z_encoded) and their corresponding mean (mu) and log variance (logvar)
            z_encoded, mu, logvar = encoder(real_B)
            #print(z_encoded.shape) #[8,8]

            # Generate fake images from real images and latent codes - vae generator
            fake_B_vae = generator(real_A, z_encoded)

            #next use clrgan generator, encoder
            z_random = torch.randn_like(z_encoded)
            fake_B_clr = generator(real_A, z_random)
            _, mu_clr, logvar_clr2 = encoder(fake_B_clr)

            #----------------------


            #generator losses -
            #1) cvaegan adversarial loss (pass vaegan's generator generated fake image to discriminator vae and calculate adv loss)
            fake_D_VAE_scores = D_VAE(fake_B_vae)
            vae_gan_loss = mse_loss(fake_D_VAE_scores, torch.ones_like(fake_D_VAE_scores))
            # we used ones_like above as generator wants to convince disc that fake images should be predicted as 1

            #2) clrgan adversarial loss - (pass clrgan generator generated fake images to disc LR and calculate adv loss)
            fake_D_LR_scores = D_LR(fake_B_clr)
            clr_gan_loss = mse_loss(fake_D_LR_scores, torch.ones_like(fake_D_LR_scores))

            #3) KL Divergence loss
            KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            #4) CVAEGAN Reconstruction Loss (L1 loss)
            cvaegan_l1 = mae_loss(fake_B_vae, real_B)

            #In all of components above, it doesnt make sense to freeze generator or encoder.
            # add all losses
            G_loss = cvaegan_l1 * lambda_pixel + KLD_loss * lambda_kl + vae_gan_loss + clr_gan_loss
            G_loss.backward(retain_graph = True)  #Here I should use retain graph
            #as below clrgan_l1 uses encoder, generator as was used by G_loss. So we should not free the computation graph


            l1_clr = mae_loss(mu_clr, z_random)
            clrgan_l1 = l1_clr * lambda_latent
            clrgan_l1.backward()

            #update optimizers
            optimizer_E.step()
            optimizer_G.step()

            #Generator, Encoder training done

            #----------------------
            #Now we train discriminator.

            optimizer_D_VAE.zero_grad()
            optimizer_D_LR.zero_grad()

            # Calculate discriminator loss for VAE-GAN
            real_D_VAE_scores = D_VAE(real_B)
            D_loss_cVAE = calculate_D_loss(fake_B_vae, D_VAE, real_B)
            D_loss_cVAE.backward()
            optimizer_D_VAE.step()

            # Calculate discriminator loss for cLR-GAN
            real_D_LR_scores = D_LR(real_B)
            D_loss_cLRGAN = calculate_D_loss(fake_B_clr, D_LR, real_B)
            D_loss_cLRGAN.backward()
            optimizer_D_LR.step()


            #add all losses
            running_total_loss += (D_loss_cVAE +D_loss_cLRGAN + cvaegan_l1*lambda_pixel + KLD_loss * lambda_kl + clrgan_l1*lambda_latent +vae_gan_loss  +clr_gan_loss).item()
            running_loss_cvaegan_l1 += cvaegan_l1.item()
            running_loss_KLD += KLD_loss.item()
            running_loss_D_loss_cVAE += D_loss_cVAE.item()
            running_loss_D_loss_cLRGAN += D_loss_cLRGAN.item()
            running_loss_clrgan_l1 += clrgan_l1.item()
            running_loss_vae_gan_loss += vae_gan_loss.item()
            running_loss_LR_GAN_loss += clr_gan_loss.item()


            step += 1

            ########## Visualization ##########
            if step % report_feq == report_feq-1:
                print('Train Epoch: {} {:.0f}% \tTotal Loss: {:.6f} \loss_cvaegan_l1: {:.6f}\loss_KLD: {:.6f}\D_loss_cVAE: {:.6f}\D_loss_cLRGAN: {:.6f}\loss_clrgan_l1: {:.6f}\VAE_GAN_loss: {:.6f}\LR_GAN_loss: {:.6f}'.format
                        (e+1, 100. * idx / len(loader), running_total_loss / report_feq,
                        running_loss_cvaegan_l1/report_feq, running_loss_KLD/report_feq,
                        running_loss_D_loss_cVAE/report_feq, running_loss_D_loss_cLRGAN/report_feq,
                        running_loss_clrgan_l1/report_feq,
                         running_loss_vae_gan_loss/report_feq,
                         running_loss_LR_GAN_loss/report_feq))

                #now store losses in list for plot
                total_loss_history.append(running_total_loss/report_feq)
                loss_cvaegan_l1_history.append(running_loss_cvaegan_l1/report_feq)
                loss_KLD_history.append(running_loss_KLD/report_feq)
                loss_D_loss_cVAE_history.append(running_loss_D_loss_cVAE/report_feq)
                loss_D_loss_cLRGAN_history.append(running_loss_D_loss_cLRGAN/report_feq)
                loss_clrgan_l1_history.append(running_loss_clrgan_l1/report_feq)
                loss_vae_gan_loss_history.append(running_loss_vae_gan_loss/report_feq)
                loss_LR_GAN_loss_history.append(running_loss_LR_GAN_loss/report_feq)

                #now reset once saved
                running_total_loss = 0
                running_loss_cvaegan_l1 = 0
                running_loss_KLD = 0
                running_loss_D_loss_cVAE = 0
                running_loss_D_loss_cLRGAN = 0
                running_loss_clrgan_l1 = 0
                running_loss_vae_gan_loss = 0
                running_loss_LR_GAN_loss = 0


                #Visualize generated images & loss
                plot_sketch_to_multimodal(generator,real_A[0],latent_dim)

                plot_loss(total_loss_history,loss_cvaegan_l1_history,
             loss_KLD_history,loss_D_loss_cVAE_history,
             loss_D_loss_cLRGAN_history,
             loss_clrgan_l1_history,
                        loss_vae_gan_loss_history,loss_LR_GAN_loss_history)

            if step % save_feq == save_feq-1:
              #Checkpointing code
              checkpoint_filename = f'checkpoint_epoch{e}_step_{step}.pt'
              checkpoint_path = os.path.join('/content/drive/MyDrive/classwork/cis680/project/checkpoints/', checkpoint_filename)
              print(f"Saving checkpoint : {checkpoint_filename} at {checkpoint_path}")

              checkpoint = {
                  #store losses list
                  "loss_total": total_loss_history,
                  "cvaegan_l1": loss_cvaegan_l1_history,
                  "KLD": loss_KLD_history,
                  "D_loss_cVAE": loss_D_loss_cVAE_history,
                  "D_loss_cLRGAN": loss_D_loss_cLRGAN_history,
                  "clrgan_l1" : loss_clrgan_l1_history,
                  "VAE_GAN_loss" : loss_vae_gan_loss_history,
                  "LR_GAN_loss" : loss_LR_GAN_loss_history,
                  #store generators, discriminators, encoders since they are getting updated in backprop
                  "generator": generator.state_dict(),
                  "encoder": encoder.state_dict(),
                  "D_VAE": D_VAE.state_dict(),
                  "D_LR": D_LR.state_dict(),
                  #store optimizer & epoch info
                  "optimizer_G" : optimizer_G.state_dict(),
                  "optimizer_E": optimizer_E.state_dict(),
                  "optimizer_D_VAE": optimizer_D_VAE.state_dict(),
                  "optimizer_D_LR" : optimizer_D_LR.state_dict(),
                  "epoch" : e

                            }

              torch.save(checkpoint, checkpoint_path)



    return total_loss_history,loss_cvaegan_l1_history,loss_KLD_history,loss_D_loss_cVAE_history,loss_D_loss_cLRGAN_history, loss_clrgan_l1_history












if __name__ == '__main__':
	train()
