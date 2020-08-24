import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#####
#   models for training
#####
class VAEGAN(nn.Module):
    def __init__(self, vae_encoder_layer =1, vae_decoder_layer = 2, z_size = 128, d_layer = 1, sample_size=64, gen_num_layer_unit = 32, dis_num_layer_unit = 16):
        super(VAEGAN, self).__init__()
        self.z_size = z_size
        #VAE
        self.VAE = VAE(vae_encoder_layer, vae_decoder_layer, z_size, sample_size=sample_size, gen_num_layer_unit = gen_num_layer_unit, dis_num_layer_unit = dis_num_layer_unit)
        #GAN
        self.discriminator = Discriminator(d_layer, z_size, input_size=sample_size, num_layer_unit=dis_num_layer_unit)

    def forward(self, x):
        #VAE
        x = self.VAE(x)
        #classifier and discriminator
        x = self.discriminator(x)
        return x

    def generate_tree(self, check_D = False, num_trees = 1, num_try = 100, config = None):
        #num_try is number of trial to generate a tree that can fool D
        #total number of sample generated = num_trees * num_try
        vae = self.VAE.to(device)
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
                z = torch.randn(batch_size, vae.z_size).float().to(device)
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
                z = torch.randn(batch_size, vae.z_size).float().to(device)
                
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


class GAN(nn.Module):
    def __init__(self, g_layer = 2, d_layer = 1, z_size=128, sample_size=64, gen_num_layer_unit = 32, dis_num_layer_unit = 16):
        super(GAN, self).__init__()
        self.generator = Generator(g_layer, z_size, output_size=sample_size, num_layer_unit=gen_num_layer_unit)
        self.discriminator = Discriminator(d_layer, z_size, input_size=sample_size, num_layer_unit=dis_num_layer_unit)

    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x

    def generate_tree(self, check_D = False, num_trees = 1, num_try = 100, config = None):
        #num_try is number of trial to generate a tree that can fool D
        #total number of sample generated = num_trees * num_try
        
        generator = self.generator.to(device)
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
                z = torch.randn(batch_size, 128).to(device)
                
                tree_fake = generator(z)[:,0,:,:,:]
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
                z = torch.randn(batch_size, 128).to(device)
                
                tree_fake = generator(z)
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

#####
#   building blocks for models 
#####

class Generator(nn.Module):
    def __init__(self, layer_per_block=2, z_size=128, output_size=64, num_layer_unit = 32):
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
            gen_module.append(nn.ReLU(True))

            for _ in range(layer_per_block-1):
                gen_module.append(nn.ConvTranspose3d(num_layer_unit2, num_layer_unit2, 3, 1, padding = 1))
                gen_module.append(nn.BatchNorm3d(num_layer_unit2))
                gen_module.append(nn.ReLU(True))

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
    def __init__(self, layer_per_block=2, z_size = 128, input_size=64, num_layer_unit = 16):
        super(Discriminator, self).__init__()

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
            dis_module.append(nn.ReLU(True))

            for _ in range(layer_per_block-1):
                dis_module.append(nn.Conv3d(num_layer_unit2, num_layer_unit2, 3, 1, padding = 1))
                dis_module.append(nn.BatchNorm3d(num_layer_unit2))
                dis_module.append(nn.ReLU(True))

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

    def forward(self, x):

        x = self.dis(x)
        x = x.view(x.shape[0], -1)
        fx = self.dis_fc1(x)
        x = self.dis_fc2(fx)
        return x, fx

class VAE(nn.Module):
    def __init__(self, vae_encoder_layer = 1, vae_decoder_layer = 2, z_size = 128, sample_size=64, gen_num_layer_unit = 32, dis_num_layer_unit = 16):
        super(VAE, self).__init__()
        self.z_size = z_size
        #VAE
        self.vae_encoder = Discriminator(vae_encoder_layer, z_size, input_size=sample_size, num_layer_unit=dis_num_layer_unit)
        self.encoder_mean = nn.Linear(z_size, z_size)
        self.encoder_logvar = nn.Linear(z_size, z_size)
        self.vae_decoder = Generator(vae_decoder_layer, z_size, output_size=sample_size, num_layer_unit=gen_num_layer_unit)


    #reference: https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner/blob/master/CVAE-GAN/CVAE-GAN.py
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        #VAE
        _, f = self.vae_encoder(x)
        x_mean = self.encoder_mean(f)
        x_logvar = self.encoder_logvar(f)
        x = self.noise_reparameterize(x_mean, x_logvar)
        x = self.vae_decoder(x)
        return x

    def generate_sample(self, z):
        x = self.vae_decoder(z)
        return x

