import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#####
#   models for training
#####
class VAEGAN(pl.LightningModule):
    def __init__(self, config):
        super(VAEGAN, self).__init__()
        # assert(vae.sample_size == discriminator.input_size)
        self.config = config
        #create components
        decoder = Generator(config.vae_decoder_layer, config.z_size, config.array_size, config.gen_num_layer_unit)
        encoder = Discriminator(config.vae_encoder_layer, config.z_size, config.array_size, config.dis_num_layer_unit)
        vae = VAE(encoder=encoder, decoder=decoder)

        discriminator = Discriminator(config.d_layer, config.z_size, config.array_size, config.dis_num_layer_unit)

        #VAE
        self.vae = vae
        #GAN
        self.discriminator = discriminator

        
    def forward(self, x):
        #VAE
        x = self.vae(x)
        #classifier and discriminator
        x = self.discriminator(x)
        return x

    def configure_optimizers(self):
        config = self.config
        vae = self.vae
        discriminator = self.discriminator

        #optimizer
        if config.vae_opt == "Adam":
            self.vae_optimizer = optim.Adam(vae.parameters(), lr=config.vae_lr)
        else:
            self.vae_optimizer = optim.SGD(vae.parameters(), lr=config.vae_lr)

        if config.dis_opt == "Adam":
            self.discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_lr)
        else:
            self.discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=config.d_lr)

        return self.vae_optimizer, self.discriminator_optimizer

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config
        vae_recon_loss_factor = config.vae_recon_loss_factor
        balance_voxel_in_space = config.balance_voxel_in_space

        dataset_batch = dataset_batch[0]    #dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch.float().to(device)

        batch_size = dataset_batch.shape[0]
        vae = self.vae
        discriminator = self.discriminator

        #loss function
        criterion_label = nn.BCELoss
        criterion_reconstruct = nn.MSELoss
        criterion_label = criterion_label(reduction='mean')
            
        #labels
        real_label = torch.unsqueeze(torch.ones(batch_size),1).float().to(device)
        fake_label = torch.unsqueeze(torch.zeros(batch_size),1).float().to(device)


        if optimizer_idx == 0:
            ############
            #   VAE
            ############
           
            reconstructed_data, mu, logVar = vae(dataset_batch, output_all=True)

            if balance_voxel_in_space:
                mask_hasvoxel = dataset_batch.clone().detach()
                mask_novoxel = torch.logical_not(mask_hasvoxel).float()
                num_hasvoxel = torch.sum(mask_hasvoxel)
                num_novoxel = torch.sum(mask_novoxel)
                total_voxel = num_hasvoxel + num_novoxel

                mask_hasvoxel = mask_hasvoxel / num_hasvoxel * total_voxel/2.
                mask_novoxel = mask_novoxel / num_novoxel * total_voxel/2.
                final_mask = mask_hasvoxel + mask_novoxel   #note that sum of final mask should be the same as the space volume
                
                #not grad on mask
                final_mask = final_mask.clone().detach().requires_grad_(False)
                vae_rec_loss = torch.mean(criterion_reconstruct(reduction='none')(reconstructed_data, dataset_batch) * final_mask)
            else:
                vae_rec_loss = criterion_reconstruct(reduction='mean')(reconstructed_data, dataset_batch)    #loss is scaled to one
            

            #add KL loss
            KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar)
            vae_rec_loss += KL

            #output of the vae should fool discriminator
            vae_out_d = discriminator(reconstructed_data)
            vae_d_loss = criterion_label(vae_out_d, real_label)

            vae_loss = (vae_recon_loss_factor * vae_rec_loss + vae_d_loss) / (vae_recon_loss_factor + 1)   #scale the loss to one

            result = pl.TrainResult(minimize=vae_loss, checkpoint_on=vae_loss)
            result.log('vae_loss', vae_loss, on_epoch=True, prog_bar=True)
            return result

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############
            
            #generate fake trees
            latent_size = vae.decoder_z_size
            z = torch.randn(batch_size, latent_size).float().to(device) #noise vector
            tree_fake = vae.generate_sample(z)

            #fake data (data from generator) 
            dout_fake = discriminator(tree_fake.clone().detach())   #detach so no update to generator
            dloss_fake = criterion_label(dout_fake, fake_label)
            #real data (data from dataloader)
            dout_real = discriminator(dataset_batch)
            dloss_real = criterion_label(dout_real, real_label)

            dloss = (dloss_fake + dloss_real) / 2   #scale the loss to one

            result = pl.TrainResult(minimize=dloss)
            result.log('dloss', dloss, on_epoch=True, prog_bar=True)
            return result


    def generate_tree(self, check_D = False, num_trees = 1, num_try = 100):
        config = self.config
        #num_try is number of trial to generate a tree that can fool D
        #total number of sample generated = num_trees * num_try
        vae = self.vae.to(device)
        discriminator = self.discriminator.to(device)

        result = None

        if config is None:
            batch_size = 4
        else:
            batch_size = config.batch_size

        if not check_D:
            num_tree_total = num_trees
            num_runs = int(np.ceil(num_tree_total / batch_size))
            #ignore discriminator
            for i in range(num_runs):
                #generate noise vector
                z = torch.randn(batch_size, vae.decoder_z_size).float().to(device)
                tree_fake = vae.generate_sample(z)[:,0,:,:,:]
                selected_trees = tree_fake.detach().cpu().numpy()
                if result is None:
                    result = selected_trees
                else:
                    result = np.concatenate((result, selected_trees), axis=0)
        else:
            num_tree_total = num_trees * num_try
            num_runs = int(np.ceil(num_tree_total / batch_size))
            #only show samples can fool discriminator
            for i in range(num_runs):
                #generate noise vector
                z = torch.randn(batch_size, vae.decoder_z_size).float().to(device)
                
                tree_fake = vae.generate_sample(z)
                dout, _ = discriminator(tree_fake)
                dout = dout > 0.5
                selected_trees = tree_fake[dout].detach().cpu().numpy()
                if result is None:
                    result = selected_trees
                else:
                    result = np.concatenate((result, selected_trees), axis=0)
        #select at most num_trees
        if result.shape[0] > num_trees:
            result = result[:num_trees]
        #in case no good result
        if result.shape[0] <= 0:
            result = np.zeros((1,64,64,64))
            result[0,0,0,0] = 1
        return result


# class GAN(nn.Module):
#     def __init__(self, generator, discriminator):
#         super(GAN, self).__init__()
#         assert(generator.output_size == discriminator.input_size)

#         self.generator = generator
#         self.discriminator = discriminator

#     def forward(self, x):
#         x = self.generator(x)
#         x = self.discriminator(x)
#         return x

    
#     def compute_loss(self, dataset_batch, criterion_label = nn.BCELoss(reduction='mean') ):
#     # this function calculate loss of the model given a dataset_batch from dataloader    
#         #loss function
#         # criterion = lambda input,target : nn.BCELoss(reduction='none')(input,target).mean()    #mean per element in each data, mean over batch 
#         # criterion = nn.BCELoss(reduction='mean')    #sum over element in each sample, mean over batch

#         batch_size = dataset_batch.shape[0]
            
#         #labels
#         real_label = torch.unsqueeze(torch.ones(batch_size),1).float().to(device)
#         fake_label = torch.unsqueeze(torch.zeros(batch_size),1).float().to(device)

#         ############
#         #   discriminator
#         ############
#         #generate fake trees
#         z = torch.randn(batch_size, 128).float().to(device) #128-d noise vector
#         tree_fake = self.generator(z)

#         #real data (data from dataloader)
#         dout_real, features_real = self.discriminator(dataset_batch, output_all=True)
#         dloss_real = criterion_label(dout_real, real_label)
#         score_real = dout_real
#         #fake data (data from generator)            
#         dout_fake = self.discriminator(tree_fake.clone().detach())   #detach so no update to generator
#         dloss_fake = criterion_label(dout_fake, fake_label)
#         score_fake = dout_fake

#         #loss function (discriminator classify real data vs generated data)
#         dloss = (dloss_real + dloss_fake)/2

#         ############
#         #   generator
#         ############

#         #tree_fake is already computed above
#         dout_fake, features_fake = self.discriminator(tree_fake, output_all=True)
#         #generator should generate trees that discriminator think they are real
#         gloss = criterion_label(dout_fake, real_label)

#         return dloss, gloss

#     def generate_tree(self, check_D = False, num_trees = 1, num_try = 100, config = None):
#         #num_try is number of trial to generate a tree that can fool D
#         #total number of sample generated = num_trees * num_try
        
#         generator = self.generator.to(device)
#         discriminator = self.discriminator.to(device)

#         result = None

#         if config is None:
#             batch_size = 4
#         else:
#             batch_size = config.batch_size

#         if not check_D:
#             num_tree_total = num_trees
#             num_runs = int(np.ceil(num_tree_total / batch_size))
#             #ignore discriminator
#             for i in range(num_runs):
#                 #generate noise vector
#                 z = torch.randn(batch_size, 128).to(device)
                
#                 tree_fake = generator(z)[:,0,:,:,:]
#                 selected_trees = tree_fake.detach().cpu().numpy()
#                 if result is None:
#                     result = selected_trees
#                 else:
#                     result = np.concatenate((result, selected_trees), axis=0)
#         else:
#             num_tree_total = num_trees * num_try
#             num_runs = int(np.ceil(num_tree_total / batch_size))
#             #only show samples can fool discriminator
#             for i in range(num_runs):
#                 #generate noise vector
#                 z = torch.randn(batch_size, 128).to(device)
                
#                 tree_fake = generator(z)
#                 dout = discriminator(tree_fake)
#                 dout = dout > 0.5
#                 selected_trees = tree_fake[dout].detach().cpu().numpy()
#                 if result is None:
#                     result = selected_trees
#                 else:
#                     result = np.concatenate((result, selected_trees), axis=0)
#         #select at most num_trees
#         if result.shape[0] > num_trees:
#             result = result[:num_trees]
#         #in case no good result
#         if result.shape[0] <= 0:
#             result = np.zeros((1,64,64,64))
#             result[0,0,0,0] = 1
#         return result

#####
#   building blocks for models 
#####

class Generator(nn.Module):
    def __init__(self, layer_per_block=2, z_size=128, output_size=64, num_layer_unit = 32, activations = nn.ReLU(True)):
        super(Generator, self).__init__()
        self.z_size = z_size

        #layer_per_block must be >= 1
        if layer_per_block < 1:
            layer_per_block = 1

        self.fc_channel = 8 #16
        self.fc_size = 4

        
        self.output_size = output_size
        #need int(output_size / self.fc_size) upsampling to increase size, so we have int(output_size / self.fc_size) + 1 block
        self.num_blocks = int(np.log2(output_size) - np.log2(self.fc_size)) + 1


        if type(num_layer_unit) is list:
            if len(num_layer_unit) != self.num_blocks:
                raise Exception("For output_size="+str(output_size)+", the list of num_layer_unit should have "+str(self.num_blocks)+" elements.")
            num_layer_unit_list = num_layer_unit
        elif type(num_layer_unit) is int:
            num_layer_unit_list = [num_layer_unit] * self.num_blocks
        else:
            raise Exception("num_layer_unit should be int of list of int.")

        # add initial self.fc_channel to num_layer_unit_list
        num_layer_unit_list = [self.fc_channel] + num_layer_unit_list
        gen_module = []
        #
        for i in range(self.num_blocks):
            num_layer_unit1, num_layer_unit2 = num_layer_unit_list[i], num_layer_unit_list[i+1]

            gen_module.append(nn.ConvTranspose3d(num_layer_unit1, num_layer_unit2, 3, 1, padding = 1))
            gen_module.append(nn.BatchNorm3d(num_layer_unit2))
            gen_module.append(activations)

            for _ in range(layer_per_block-1):
                gen_module.append(nn.ConvTranspose3d(num_layer_unit2, num_layer_unit2, 3, 1, padding = 1))
                gen_module.append(nn.BatchNorm3d(num_layer_unit2))
                gen_module.append(activations)

            gen_module.append(nn.Upsample(scale_factor=2, mode='trilinear'))

        #remove extra pool layer
        gen_module = gen_module[:-1]

        #add final sigmoid 
        gen_module.append(nn.ConvTranspose3d(num_layer_unit_list[-1], 1, 3, 1, padding = 1))
        gen_module.append(nn.Sigmoid())

        

        self.gen_fc = nn.Linear(self.z_size, self.fc_channel * self.fc_size * self.fc_size * self.fc_size)
        self.gen = nn.Sequential(*gen_module)

    def forward(self, x):
        x = self.gen_fc(x)
        x = x.view(x.shape[0], self.fc_channel, self.fc_size, self.fc_size, self.fc_size)
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, layer_per_block=2, z_size = 128, input_size=64, num_layer_unit = 16, activations = nn.ReLU(True)):
        super(Discriminator, self).__init__()

        self.z_size = z_size

         #layer_per_block must be >= 1
        if layer_per_block < 1:
            layer_per_block = 1

        self.fc_size = 4    #final height of the volume in conv layers before flatten

        self.input_size = input_size
        #need int(input_size / self.fc_size) upsampling to increase size, so we have int(input_size / self.fc_size) + 1 block
        self.num_blocks = int(np.log2(input_size) - np.log2(self.fc_size)) + 1


        if type(num_layer_unit) is list:
            if len(num_layer_unit) != self.num_blocks:
                raise Exception("For input_size="+str(input_size)+", the list of num_layer_unit should have "+str(self.num_blocks)+" elements.")
            num_layer_unit_list = num_layer_unit
        elif type(num_layer_unit) is int:
            num_layer_unit_list = [num_layer_unit] * self.num_blocks
        else:
            raise Exception("num_layer_unit should be int of list of int.")

        # add initial num_unit to num_layer_unit_list
        num_layer_unit_list = [1] + num_layer_unit_list
        dis_module = []
        #5 blocks (need 4 pool to reduce size)
        for i in range(self.num_blocks):
            num_layer_unit1, num_layer_unit2 = num_layer_unit_list[i], num_layer_unit_list[i+1]

            dis_module.append(nn.Conv3d(num_layer_unit1, num_layer_unit2, 3, 1, padding = 1))
            dis_module.append(nn.BatchNorm3d(num_layer_unit2))
            dis_module.append(activations)

            for _ in range(layer_per_block-1):
                dis_module.append(nn.Conv3d(num_layer_unit2, num_layer_unit2, 3, 1, padding = 1))
                dis_module.append(nn.BatchNorm3d(num_layer_unit2))
                dis_module.append(activations)

            dis_module.append(nn.MaxPool3d((2, 2, 2)))

        #remove extra pool layer
        dis_module = dis_module[:-1]

        
        self.dis = nn.Sequential(*dis_module)

        self.dis_fc1 = nn.Sequential(
            nn.Linear(num_layer_unit_list[-1] * self.fc_size * self.fc_size * self.fc_size, z_size),
            nn.ReLU(True)
        )
        self.dis_fc2 = nn.Sequential(
            nn.Linear(z_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x, output_all=False):

        x = self.dis(x)
        x = x.view(x.shape[0], -1)
        fx = self.dis_fc1(x)
        x = self.dis_fc2(fx)
        if output_all:
            return x, fx
        else:
            return x

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        assert(encoder.input_size == decoder.output_size)
        self.sample_size = decoder.output_size
        self.encoder_z_size = encoder.z_size
        self.decoder_z_size = decoder.z_size
        #VAE
        self.vae_encoder = encoder
        self.encoder_mean = nn.Linear(self.encoder_z_size, self.decoder_z_size)
        self.encoder_logvar = nn.Linear(self.encoder_z_size, self.decoder_z_size)
        self.vae_decoder = decoder


    #reference: https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner/blob/master/CVAE-GAN/CVAE-GAN.py
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x, output_all=False):
        #VAE
        _, f = self.vae_encoder(x, output_all=True)
        x_mean = self.encoder_mean(f)
        x_logvar = self.encoder_logvar(f)
        x = self.noise_reparameterize(x_mean, x_logvar)
        x = self.vae_decoder(x)
        if output_all:
            return x, x_mean, x_logvar
        else:
            return x

    def generate_sample(self, z):
        x = self.vae_decoder(z)
        return x

