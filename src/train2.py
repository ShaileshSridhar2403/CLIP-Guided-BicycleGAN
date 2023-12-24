def train2():
        epoch_id = "cuda:0"
        checkpoints_path = 'checkpoints/'
        imgs_path = 'figures/'
        # if IN_COLAB:
        #     checkpoints_path = os.path.join(COLAB_ROOT, checkpoints_path)
        #     imgs_path = os.path.join(COLAB_ROOT, imgs_path)
        os.makedirs(checkpoints_path, exist_ok=True)
        os.makedirs(imgs_path, exist_ok=True)

        img_dir = "/content/train/"
        img_shape = (3, 128, 128)  # Please use this image dimension faster training purpose
        n_residual_blocks = 6
        num_epochs = 20
        batch_size = 8
        lr_rate = 2e-4	      # Adam optimizer learning rate
        betas = (0.5, 0.999)    # Adam optimizer beta 1, beta 2
        lambda_pixel = 10      # Loss weights for pixel loss
        lambda_latent = 0.5    # Loss weights for latent regression
        lambda_kl = 0.01        # Loss weights for kl divergence
        latent_dim = 8      # latent dimension for the encoded images from domain B
        gpu_id = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Reparameterization helper function
        # (You may need this helper function here or inside models.py, depending on your encoder
        #   implementation)

        # Random seeds (optional)
        torch.manual_seed(1)
        np.random.seed(1)

        # Define DataLoader
        dataset = Edge2Shoe(img_dir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        # Loss functions
        l1_loss = torch.nn.L1Loss().to(gpu_id)
        mse_loss = torch.nn.MSELoss().to(gpu_id)

        # Define generator, encoder and discriminators
        generator = ResNetGenerator(latent_dim, img_shape, n_residual_blocks, device=gpu_id).to(gpu_id)
        encoder = Encoder(latent_dim).to(gpu_id)
        # discriminator = PatchGANDiscriminator(img_shape).to(gpu_id)
        discriminator = Discriminator().to(gpu_id)

        # init weights
        # generator.apply(weights_init_normal)
        # encoder.apply(weights_init_normal)
        # discriminator.apply(weights_init_normal)

        # Define optimizers for networks
        optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=betas)
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=betas)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=betas)

        # For adversarial loss (optional to use)
        valid = 1
        fake = 0

        # Train loss list
        list_vae_G_train_loss = []
        list_clr_G_train_loss = []
        list_kld_train_loss = []
        list_img_train_loss = []
        list_G_train_loss = []
        list_latent_train_loss = []
        list_vae_D_train_loss = []
        list_clr_D_train_loss = []

        # Training
        total_steps = len(loader) * num_epochs
        step = 0
        print("------------------------------- Starting epoch ", str(epoch_id), "---------------------------------")
        avg_vae_G_train_loss = 0
        avg_clr_G_train_loss = 0
        avg_kld_train_loss = 0
        avg_img_train_loss = 0
        avg_G_train_loss = 0
        avg_latent_train_loss = 0
        avg_vae_D_train_loss = 0
        avg_clr_D_train_loss = 0
        count = 0

        start = time.time()
        for idx, data in enumerate(loader):
            count += 1
            # ######## Process Inputs ##########
            edge_tensor, rgb_tensor = data
            edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
            real_A = edge_tensor
            real_B = rgb_tensor

            # Adversarial ground truths
            valid = torch.Tensor(np.ones((real_A.size(0), *discriminator.output_shape))).to(gpu_id)
            valid.requires_grad = False
                             
            fake = torch.Tensor(np.zeros((real_A.size(0), *discriminator.output_shape))).to(gpu_id)
            fake.required_grad = False
                            

            # -------------------------------
            #  Forward ALL
            # ------------------------------
            encoder.train()
            generator.train()

            z_encoded,z_mu, z_logvar = encoder.forward(rgb_tensor)
            # z_encoded = reparameterization(z_mu, z_logvar, device=gpu_id)

            fake_B_encoded = generator.forward(real_A, z_encoded)

            z_random = torch.randn(real_A.shape[0], latent_dim).to(gpu_id)
            fake_B_random = generator.forward(real_A, z_random)

            z_pred,z_mu_predict, z_logvar_predict = encoder.forward(fake_B_random)

            # -------------------------------
            #  Train Generator and Encoder
            # ------------------------------
            for param in discriminator.parameters():
                param.requires_grad = False

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            # G(A) should fool D
            fake_B_encoded_label = discriminator.forward(fake_B_encoded)
            vae_G_loss = mse_loss(fake_B_encoded_label, valid)
            fake_B_random_label = discriminator.forward(fake_B_random)
            clr_G_loss = mse_loss(fake_B_random_label, valid)

            # compute KLD loss
            kld_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            # kld_loss = loss_KLD(z_mu, z_logvar, device=gpu_id)

            # Compute L1 image loss
            img_loss = l1_loss(fake_B_encoded, real_B)

            loss_G = vae_G_loss + clr_G_loss + lambda_kl * kld_loss + lambda_pixel * img_loss
            loss_G.backward(retain_graph=True)

            #  Backward Latent space
            for param in encoder.parameters():
                param.requires_grad = False

            latent_loss = l1_loss(z_mu_predict, z_random) * lambda_latent
            latent_loss.backward()

            for param in encoder.parameters():
                param.requires_grad = True

            optimizer_E.step()
            optimizer_G.step()

            # -------------------------------
            #  Train Discriminator
            # ------------------------------
            for param in discriminator.parameters():
                param.requires_grad = True

            optimizer_D.zero_grad()

            # Compute VAE-GAN discriminator loss
            vae_D_loss = loss_discriminator(
                fake_B_encoded, discriminator, real_B, valid, fake, mse_loss)
            vae_D_loss.backward()

            clr_D_loss = loss_discriminator(
                fake_B_random, discriminator, real_B, valid, fake, mse_loss)
            clr_D_loss.backward()

            optimizer_D.step()

            # -------------------------------
            #  Add up loss
            # ------------------------------
            avg_vae_G_train_loss += vae_G_loss.item()
            avg_clr_G_train_loss += clr_G_loss.item()
            avg_kld_train_loss += kld_loss.item()
            avg_img_train_loss += img_loss.item()
            avg_G_train_loss += loss_G.item()
            avg_latent_train_loss += latent_loss.item()
            avg_vae_D_train_loss += vae_D_loss.item()
            avg_clr_D_train_loss += clr_D_loss.item()

            print("epoch {} iter {}; loss_G: {:.4f}; loss_D: {:.4f}; latent: {:.4f}; KLD: {:.4f}".format(
                epoch_id, idx, loss_G.item(), vae_D_loss.item() + clr_D_loss.item(), latent_loss.item(), kld_loss.item()))

            if (idx + 1) % (len(loader) // 5) == 0:
                # -------------------------------
                #  Save model
                # ------------------------------
                path = os.path.join(checkpoints_path,
                                    'bicycleGAN_epoch_' + str(epoch_id) + '_' + str(idx))
                torch.save({
                    'epoch': epoch_id,
                    'encoder_state_dict': encoder.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_E': optimizer_E.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict()
                }, path)

                # -------------------------------
                #  Visualization
                # ------------------------------
            if (idx + 1) % 100 == 0:
                vis_fake_B_encoded = denorm(fake_B_encoded[0].detach()).cpu().data.numpy().astype(np.uint8)
                vis_fake_B_random = denorm(fake_B_random[0].detach()).cpu().data.numpy().astype(np.uint8)
                vis_real_B = denorm(real_B[0].detach()).cpu().data.numpy().astype(np.uint8)
                vis_real_A = denorm(real_A[0].detach()).cpu().data.numpy().astype(np.uint8)
                fig, axs = plt.subplots(2, 2, figsize=(5, 5))

                axs[0, 0].imshow(vis_real_A.transpose(1, 2, 0))
                axs[0, 0].set_title('real images')
                axs[0, 1].imshow(vis_fake_B_encoded.transpose(1, 2, 0))
                axs[0, 1].set_title('generated images')
                axs[1, 0].imshow(vis_real_B.transpose(1, 2, 0))
                axs[1, 1].imshow(vis_fake_B_random.transpose(1, 2, 0))
                plt.show()
                # path = os.path.join(imgs_path, 'epoch_' + str(epoch_id) + '_' + str(idx) + '.png')
                # plt.savefig(path)

        # -------------------------------
        #  Main Storage
        # ------------------------------
        list_vae_G_train_loss.append(avg_vae_G_train_loss / count)
        list_clr_G_train_loss.append(avg_clr_G_train_loss / count)
        list_kld_train_loss.append(avg_kld_train_loss / count)
        list_img_train_loss.append(avg_img_train_loss / count)
        list_G_train_loss.append(avg_G_train_loss / count)
        list_latent_train_loss.append(avg_latent_train_loss / count)
        list_vae_D_train_loss.append(avg_vae_D_train_loss / count)
        list_clr_D_train_loss.append(avg_clr_D_train_loss / count)