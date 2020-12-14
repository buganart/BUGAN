from bugan.functionsPL import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from argparse import Namespace, ArgumentParser
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


#####
#   models for training
#####
class VAE_train(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # log argument
        parser.add_argument("--log_interval", type=int, default=10)
        parser.add_argument("--log_num_samples", type=int, default=1)

        # model specific argument (VAE)
        parser.add_argument("--z_size", type=int, default=128)
        parser.add_argument("--resolution", type=int, default=resolution)
        # number of layer per block
        parser.add_argument("--vae_decoder_layer", type=int, default=2)
        parser.add_argument("--vae_encoder_layer", type=int, default=1)
        # optimizer in {"Adam", "SGD"}
        parser.add_argument("--vae_opt", type=str, default="Adam")
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--rec_loss", type=str, default="MSELoss")
        # activation default leakyReLU
        parser.add_argument("--activation_leakyReLU_slope", type=float, default=0.01)
        # Dropout probability
        parser.add_argument("--dropout_prob", type=float, default=0.3)
        # instance noise (add noise to real data/generate data)
        parser.add_argument(
            "--linear_annealed_instance_noise_epoch", type=int, default=2000
        )
        parser.add_argument("--instance_noise", type=float, default=0.1)
        # spectral_norm
        parser.add_argument("--spectral_norm", type=bool, default=False)
        # learning rate
        parser.add_argument("--vae_lr", type=float, default=0.0025)
        # number of unit per layer
        if resolution == 32:
            decoder_num_layer_unit = [1024, 512, 256, 128]
            encoder_num_layer_unit = [32, 64, 128, 128]
        else:
            decoder_num_layer_unit = [1024, 512, 256, 128, 128]
            encoder_num_layer_unit = [32, 64, 64, 128, 128]
        parser.add_argument("--decoder_num_layer_unit", default=decoder_num_layer_unit)
        parser.add_argument("--encoder_num_layer_unit", default=encoder_num_layer_unit)

        return parser

    def __init__(self, config):
        super(VAE_train, self).__init__()
        # assert(vae.sample_size == discriminator.input_size)
        self.config = config
        self.save_hyperparameters("config")
        # add missing default parameters
        parser = self.add_model_specific_args(
            ArgumentParser(), resolution=config.resolution
        )
        args = parser.parse_args([])
        config = combine_namespace(args, config)
        self.config = config
        # create components
        decoder = Generator(
            config.vae_decoder_layer,
            config.z_size,
            config.resolution,
            config.decoder_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )
        encoder = Discriminator(
            config.vae_encoder_layer,
            config.z_size,
            config.resolution,
            config.encoder_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )
        vae = VAE(encoder=encoder, decoder=decoder)

        # VAE
        self.vae = vae

        # others
        self.noise_magnitude = config.instance_noise

    def configure_optimizers(self):
        config = self.config
        vae = self.vae

        # optimizer
        self.vae_optimizer = get_model_optimizer(vae, config.vae_opt, config.vae_lr)

        if hasattr(config, "cyclicLR_magnitude"):
            vae_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.vae_optimizer,
                base_lr=config.vae_lr / config.cyclicLR_magnitude,
                max_lr=config.vae_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            return [self.vae_optimizer], [vae_scheduler]
        else:
            return self.vae_optimizer

    def on_train_epoch_start(self):
        # reset ep_loss
        self.vae_ep_loss = []

        # calc instance noise
        if self.noise_magnitude > 0:
            # linear annealed noise
            noise_rate = (
                self.config.linear_annealed_instance_noise_epoch - self.current_epoch
            ) / self.config.linear_annealed_instance_noise_epoch
            self.noise_magnitude = self.config.instance_noise * noise_rate
        else:
            self.noise_magnitude = 0

        # set model to train
        self.vae.train()

    def on_train_epoch_end(self, epoch_output):

        # save model if necessary
        log_dict = {"VAE loss": np.mean(self.vae_ep_loss), "epoch": self.current_epoch}

        log_media = (
            self.current_epoch % self.config.log_interval == 0
        )  # boolean whether to log image/3D object

        wandbLog(
            self,
            log_dict,
            log_media=log_media,
            log_num_samples=self.config.log_num_samples,
        )

    def training_step(self, dataset_batch, batch_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0]
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add instance noise
        instance_noise = self.generate_noise_for_samples(dataset_batch)
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch, instance_noise)

        dataset_batch = dataset_batch.float()

        batch_size = dataset_batch.shape[0]
        vae = self.vae

        # loss function
        criterion_reconstruct = get_loss_function_with_logit(config.rec_loss)

        ############
        #   VAE
        ############

        reconstructed_data, mu, logVar = vae(dataset_batch, output_all=True)
        # add instance noise
        reconstructed_data = self.add_noise_to_samples(
            reconstructed_data, instance_noise
        )

        vae_rec_loss = criterion_reconstruct(reconstructed_data, dataset_batch)

        # add KL loss
        KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1.0 - logVar)
        vae_rec_loss += KL

        vae_loss = vae_rec_loss

        # record loss
        self.vae_ep_loss.append(vae_loss.detach().cpu().numpy())

        return vae_loss

    def generate_noise_for_samples(self, data):
        if self.noise_magnitude <= 0:
            return 0
        # create uniform noise
        noise = torch.rand(data.shape) * 2 - 1
        noise = self.noise_magnitude * noise  # noise in [-magn, magn]
        noise = noise.float().type_as(data).detach()

        return noise

    def add_noise_to_samples(self, data, noise):
        if self.noise_magnitude <= 0:
            return data
        # add instance noise
        # now batch in [-1+magn, 1-magn]
        data = data * (1 - self.noise_magnitude)
        data = data + noise
        return data

    def generate_tree(self, num_trees=1):
        config = self.config
        generator = self.vae.vae_decoder

        return generate_tree(
            generator,
            config.resolution,
            num_trees=num_trees,
            batch_size=config.batch_size,
        )


class VAEGAN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # log argument
        parser.add_argument("--log_interval", type=int, default=10)
        parser.add_argument("--log_num_samples", type=int, default=1)

        # model specific argument (VAE, discriminator)
        parser.add_argument("--z_size", type=int, default=128)
        parser.add_argument("--resolution", type=int, default=resolution)
        # number of layer per block
        parser.add_argument("--vae_decoder_layer", type=int, default=2)
        parser.add_argument("--vae_encoder_layer", type=int, default=1)
        parser.add_argument("--d_layer", type=int, default=1)
        # optimizer in {"Adam", "SGD"}
        parser.add_argument("--vae_opt", type=str, default="Adam")
        parser.add_argument("--dis_opt", type=str, default="Adam")
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--label_loss", type=str, default="BCELoss")
        parser.add_argument("--rec_loss", type=str, default="MSELoss")
        # activation default leakyReLU
        parser.add_argument("--activation_leakyReLU_slope", type=float, default=0.01)
        # Dropout probability
        parser.add_argument("--dropout_prob", type=float, default=0.3)
        # real/fake label flip probability
        parser.add_argument("--label_flip_prob", type=float, default=0.1)
        # real/fake label noise magnitude
        parser.add_argument("--label_noise", type=float, default=0.2)
        # instance noise (add noise to real data/generate data)
        parser.add_argument(
            "--linear_annealed_instance_noise_epoch", type=int, default=2000
        )
        parser.add_argument("--instance_noise", type=float, default=0.1)
        # spectral_norm
        parser.add_argument("--spectral_norm", type=bool, default=False)
        # learning rate
        parser.add_argument("--vae_lr", type=float, default=0.0025)
        parser.add_argument("--d_lr", type=float, default=0.00005)
        # number of unit per layer
        if resolution == 32:
            decoder_num_layer_unit = [1024, 512, 256, 128]
            encoder_num_layer_unit = [32, 64, 128, 128]
            dis_num_layer_unit = [32, 64, 128, 128]
        else:
            decoder_num_layer_unit = [1024, 512, 256, 128, 128]
            encoder_num_layer_unit = [32, 64, 64, 128, 128]
            dis_num_layer_unit = [32, 64, 64, 128, 128]
        parser.add_argument("--decoder_num_layer_unit", default=decoder_num_layer_unit)
        parser.add_argument("--encoder_num_layer_unit", default=encoder_num_layer_unit)
        parser.add_argument("--dis_num_layer_unit", default=dis_num_layer_unit)

        return parser

    def __init__(self, config):
        super(VAEGAN, self).__init__()
        # assert(vae.sample_size == discriminator.input_size)
        self.config = config
        self.save_hyperparameters("config")
        # add missing default parameters
        parser = self.add_model_specific_args(
            ArgumentParser(), resolution=config.resolution
        )
        args = parser.parse_args([])
        config = combine_namespace(args, config)
        self.config = config
        # create components
        decoder = Generator(
            config.vae_decoder_layer,
            config.z_size,
            config.resolution,
            config.decoder_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )
        encoder = Discriminator(
            config.vae_encoder_layer,
            config.z_size,
            config.resolution,
            config.encoder_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )
        vae = VAE(encoder=encoder, decoder=decoder)

        discriminator = Discriminator(
            config.d_layer,
            config.z_size,
            config.resolution,
            config.dis_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        # VAE
        self.vae = vae
        # GAN
        self.discriminator = discriminator

        # others
        self.noise_magnitude = config.instance_noise

    def forward(self, x):
        # VAE
        x = self.vae(x)
        x = F.tanh(x)
        # classifier and discriminator
        x = self.discriminator(x)
        return x

    def configure_optimizers(self):
        config = self.config
        vae = self.vae
        discriminator = self.discriminator

        # optimizer
        self.vae_optimizer = get_model_optimizer(vae, config.vae_opt, config.vae_lr)
        self.discriminator_optimizer = get_model_optimizer(
            discriminator, config.dis_opt, config.d_lr
        )
        if hasattr(config, "cyclicLR_magnitude"):
            vae_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.vae_optimizer,
                base_lr=config.vae_lr / config.cyclicLR_magnitude,
                max_lr=config.vae_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            d_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.discriminator_optimizer,
                base_lr=config.d_lr / config.cyclicLR_magnitude,
                max_lr=config.d_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            return [self.vae_optimizer, self.discriminator_optimizer], [
                vae_scheduler,
                d_scheduler,
            ]
        else:
            return self.vae_optimizer, self.discriminator_optimizer

    def on_train_epoch_start(self):
        # reset ep_loss
        self.d_ep_loss = []
        self.vae_ep_loss = []

        # calc instance noise
        if self.noise_magnitude > 0:
            # linear annealed noise
            noise_rate = (
                self.config.linear_annealed_instance_noise_epoch - self.current_epoch
            ) / self.config.linear_annealed_instance_noise_epoch
            self.noise_magnitude = self.config.instance_noise * noise_rate
        else:
            self.noise_magnitude = 0

        # set model to train
        self.vae.train()
        self.discriminator.train()

    def on_train_epoch_end(self, epoch_output):

        # save model if necessary
        log_dict = {
            "discriminator loss": np.mean(self.d_ep_loss),
            "VAE loss": np.mean(self.vae_ep_loss),
            "epoch": self.current_epoch,
        }

        log_media = (
            self.current_epoch % self.config.log_interval == 0
        )  # boolean whether to log image/3D object

        wandbLog(
            self,
            log_dict,
            log_media=log_media,
            log_num_samples=self.config.log_num_samples,
        )

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0]
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add instance noise
        instance_noise = self.generate_noise_for_samples(dataset_batch)
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch, instance_noise)

        dataset_batch = dataset_batch.float()

        batch_size = dataset_batch.shape[0]
        vae = self.vae
        discriminator = self.discriminator

        # loss function
        criterion_label = get_loss_function_with_logit(config.label_loss)
        criterion_reconstruct = get_loss_function_with_logit(config.rec_loss)

        # labels
        # soft label
        # modified scale to [1-label_noise,1]
        # modified scale to [0,label_noise]
        real_label = 1 - (torch.rand(batch_size) * config.label_noise)
        fake_label = torch.rand(batch_size) * config.label_noise
        # add noise to label
        # P(label_flip_prob) label flipped
        label_flip_mask = torch.bernoulli(
            torch.ones(batch_size) * config.label_flip_prob
        )
        real_label = (1 - label_flip_mask) * real_label + label_flip_mask * (
            1 - real_label
        )
        fake_label = (1 - label_flip_mask) * fake_label + label_flip_mask * (
            1 - fake_label
        )

        real_label = torch.unsqueeze(real_label, 1).float().type_as(dataset_batch)
        fake_label = torch.unsqueeze(fake_label, 1).float().type_as(dataset_batch)

        if optimizer_idx == 0:
            ############
            #   VAE
            ############

            reconstructed_data, mu, logVar = vae(dataset_batch, output_all=True)
            # add noise to data
            reconstructed_data = self.add_noise_to_samples(
                reconstructed_data, instance_noise
            )

            vae_rec_loss = criterion_reconstruct(reconstructed_data, dataset_batch)

            # add KL loss
            KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1.0 - logVar)
            vae_rec_loss += KL

            # output of the vae should fool discriminator
            vae_out_d = discriminator(F.tanh(reconstructed_data))
            vae_d_loss = criterion_label(vae_out_d, real_label)

            vae_loss = (vae_rec_loss + vae_d_loss) / 2

            # record loss
            self.vae_ep_loss.append(vae_loss.detach().cpu().numpy())

            return vae_loss

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############

            # generate fake trees
            latent_size = vae.decoder_z_size
            z = (
                torch.randn(batch_size, latent_size).float().type_as(dataset_batch)
            )  # noise vector
            tree_fake = F.tanh(vae.generate_sample(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # fake data (data from generator)
            dout_fake = discriminator(
                tree_fake.clone().detach()
            )  # detach so no update to generator
            dloss_fake = criterion_label(dout_fake, fake_label)
            # real data (data from dataloader)
            dout_real = discriminator(dataset_batch)
            dloss_real = criterion_label(dout_real, real_label)

            dloss = (dloss_fake + dloss_real) / 2  # scale the loss to one

            # record loss
            self.d_ep_loss.append(dloss.detach().cpu().numpy())

            return dloss

    def generate_noise_for_samples(self, data):
        if self.noise_magnitude <= 0:
            return 0
        # create uniform noise
        noise = torch.rand(data.shape) * 2 - 1
        noise = self.noise_magnitude * noise  # noise in [-magn, magn]
        noise = noise.float().type_as(data).detach()

        return noise

    def add_noise_to_samples(self, data, noise):
        if self.noise_magnitude <= 0:
            return data
        # add instance noise
        # now batch in [-1+magn, 1-magn]
        data = data * (1 - self.noise_magnitude)
        data = data + noise
        return data

    def generate_tree(self, num_trees=1):
        config = self.config
        generator = self.vae.vae_decoder

        return generate_tree(
            generator,
            config.resolution,
            num_trees=num_trees,
            batch_size=config.batch_size,
        )


class GAN(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # log argument
        parser.add_argument("--log_interval", type=int, default=10)
        parser.add_argument("--log_num_samples", type=int, default=1)

        # model specific argument (Generator, discriminator)
        parser.add_argument("--z_size", type=int, default=128)
        parser.add_argument("--resolution", type=int, default=resolution)
        # number of layer per block
        parser.add_argument("--g_layer", type=int, default=2)
        parser.add_argument("--d_layer", type=int, default=1)
        # optimizer in {"Adam", "SGD"}
        parser.add_argument("--gen_opt", type=str, default="Adam")
        parser.add_argument("--dis_opt", type=str, default="Adam")
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--label_loss", type=str, default="BCELoss")
        # activation default leakyReLU
        parser.add_argument("--activation_leakyReLU_slope", type=float, default=0.01)
        # Dropout probability
        parser.add_argument("--dropout_prob", type=float, default=0.3)
        # real/fake label flip probability
        parser.add_argument("--label_flip_prob", type=float, default=0.1)
        # real/fake label noise magnitude
        parser.add_argument("--label_noise", type=float, default=0.2)
        # instance noise (add noise to real data/generate data)
        parser.add_argument(
            "--linear_annealed_instance_noise_epoch", type=int, default=2000
        )
        parser.add_argument("--instance_noise", type=float, default=0.1)
        # spectral_norm
        parser.add_argument("--spectral_norm", type=bool, default=False)
        # learning rate
        parser.add_argument("--g_lr", type=float, default=0.0025)
        parser.add_argument("--d_lr", type=float, default=0.00005)
        # number of unit per layer
        if resolution == 32:
            gen_num_layer_unit = [1024, 512, 256, 128]
            dis_num_layer_unit = [32, 64, 128, 128]
        else:
            gen_num_layer_unit = [1024, 512, 256, 128, 128]
            dis_num_layer_unit = [32, 64, 64, 128, 128]
        parser.add_argument("--gen_num_layer_unit", default=gen_num_layer_unit)
        parser.add_argument("--dis_num_layer_unit", default=dis_num_layer_unit)

        return parser

    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config
        self.save_hyperparameters("config")
        # add missing default parameters
        parser = self.add_model_specific_args(
            ArgumentParser(), resolution=config.resolution
        )
        args = parser.parse_args([])
        config = combine_namespace(args, config)
        self.config = config

        # create components
        generator = Generator(
            config.g_layer,
            config.z_size,
            config.resolution,
            config.gen_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        discriminator = Discriminator(
            config.d_layer,
            config.z_size,
            config.resolution,
            config.dis_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        # GAN
        self.generator = generator
        self.discriminator = discriminator

        # others
        self.noise_magnitude = config.instance_noise

    def forward(self, x):
        # classifier and discriminator
        x = self.generator(x)
        x = F.tanh(x)
        x = self.discriminator(x)
        return x

    def configure_optimizers(self):
        config = self.config
        generator = self.generator
        discriminator = self.discriminator

        # optimizer
        self.generator_optimizer = get_model_optimizer(
            generator, config.gen_opt, config.g_lr
        )
        self.discriminator_optimizer = get_model_optimizer(
            discriminator, config.dis_opt, config.d_lr
        )

        if hasattr(config, "cyclicLR_magnitude"):
            g_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.generator_optimizer,
                base_lr=config.g_lr / config.cyclicLR_magnitude,
                max_lr=config.g_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            d_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.discriminator_optimizer,
                base_lr=config.d_lr / config.cyclicLR_magnitude,
                max_lr=config.d_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            return [self.generator_optimizer, self.discriminator_optimizer], [
                g_scheduler,
                d_scheduler,
            ]
        else:
            return self.generator_optimizer, self.discriminator_optimizer

    def on_train_epoch_start(self):
        # reset ep_loss
        self.d_ep_loss = []
        self.g_ep_loss = []

        # calc instance noise
        if self.noise_magnitude > 0:
            # linear annealed noise
            noise_rate = (
                self.config.linear_annealed_instance_noise_epoch - self.current_epoch
            ) / self.config.linear_annealed_instance_noise_epoch
            self.noise_magnitude = self.config.instance_noise * noise_rate
        else:
            self.noise_magnitude = 0

        # set model to train
        self.generator.train()
        self.discriminator.train()

    def on_train_epoch_end(self, epoch_output):

        # save model if necessary
        log_dict = {
            "discriminator loss": np.mean(self.d_ep_loss),
            "generator loss": np.mean(self.g_ep_loss),
            "epoch": self.current_epoch,
        }

        log_media = (
            self.current_epoch % self.config.log_interval == 0
        )  # boolean whether to log image/3D object

        wandbLog(
            self,
            log_dict,
            log_media=log_media,
            log_num_samples=self.config.log_num_samples,
        )

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0]
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add instance noise
        instance_noise = self.generate_noise_for_samples(dataset_batch)
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch, instance_noise)

        dataset_batch = dataset_batch.float()

        batch_size = dataset_batch.shape[0]
        generator = self.generator
        discriminator = self.discriminator

        # loss function
        criterion_label = get_loss_function_with_logit(config.label_loss)

        # labels
        # soft label
        # modified scale to [1-label_noise,1]
        # modified scale to [0,label_noise]
        real_label = 1 - (torch.rand(batch_size) * config.label_noise)
        fake_label = torch.rand(batch_size) * config.label_noise
        # add noise to label
        # P(label_flip_prob) label flipped
        label_flip_mask = torch.bernoulli(
            torch.ones(batch_size) * config.label_flip_prob
        )
        real_label = (1 - label_flip_mask) * real_label + label_flip_mask * (
            1 - real_label
        )
        fake_label = (1 - label_flip_mask) * fake_label + label_flip_mask * (
            1 - fake_label
        )

        real_label = torch.unsqueeze(real_label, 1).float().type_as(dataset_batch)
        fake_label = torch.unsqueeze(fake_label, 1).float().type_as(dataset_batch)
        if optimizer_idx == 0:
            ############
            #   generator
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            tree_fake = F.tanh(self.generator(z))

            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # tree_fake is already computed above
            dout_fake = self.discriminator(tree_fake, output_all=False)
            # generator should generate trees that discriminator think they are real
            gloss = criterion_label(dout_fake, real_label)

            # record loss
            self.g_ep_loss.append(gloss.detach().cpu().numpy())

            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            tree_fake = F.tanh(self.generator(z))

            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch, output_all=False)
            dloss_real = criterion_label(dout_real, real_label)

            # fake data (data from generator)
            dout_fake = self.discriminator(
                tree_fake.clone().detach()
            )  # detach so no update to generator
            dloss_fake = criterion_label(dout_fake, fake_label)

            # loss function (discriminator classify real data vs generated data)
            dloss = (dloss_real + dloss_fake) / 2

            # record loss
            self.d_ep_loss.append(dloss.detach().cpu().numpy())

            return dloss

    def generate_noise_for_samples(self, data):
        if self.noise_magnitude <= 0:
            return 0
        # create uniform noise
        noise = torch.rand(data.shape) * 2 - 1
        noise = self.noise_magnitude * noise  # noise in [-magn, magn]
        noise = noise.float().type_as(data).detach()

        return noise

    def add_noise_to_samples(self, data, noise):
        if self.noise_magnitude <= 0:
            return data
        # add instance noise
        # now batch in [-1+magn, 1-magn]
        data = data * (1 - self.noise_magnitude)
        data = data + noise
        return data

    def generate_tree(self, num_trees=1):
        config = self.config
        generator = self.generator

        return generate_tree(
            generator,
            config.resolution,
            num_trees=num_trees,
            batch_size=config.batch_size,
        )


class VAEGAN_Wloss_GP(VAEGAN):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # important argument
        parser.add_argument("--gp_epsilon", type=int, default=5)

        return VAEGAN.add_model_specific_args(parser, resolution)

    def gradient_penalty(self, real_tree, generated_tree):
        batch_size = real_tree.shape[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size).reshape((batch_size, 1, 1, 1, 1)).float()
        # alpha = alpha.expand_as(real_data)
        # if self.use_cuda:
        #     alpha = alpha.cuda()
        interpolated = alpha * real_tree + (1 - alpha) * generated_tree
        interpolated = interpolated.requires_grad_().float()

        # calculate prob of interpolated trees
        prob_interpolated = self.discriminator(interpolated)

        # calculate grad of prob
        grad = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()),
            create_graph=True,
            retain_graph=True,
        )[0]

        # grad have shape same as input (batch_size,1,h,w,d),
        grad = grad.view(batch_size, -1)

        # calculate norm and add epsilon to prevent sqrt(0)
        grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1e-10)

        # return gradient penalty
        return self.config.gp_epsilon * ((grad_norm - 1) ** 2).mean()

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config
        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0]
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add instance noise
        instance_noise = self.generate_noise_for_samples(dataset_batch)
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch, instance_noise)

        dataset_batch = dataset_batch.float()

        batch_size = dataset_batch.shape[0]
        vae = self.vae
        discriminator = self.discriminator

        # loss function
        criterion_label = get_loss_function_with_logit(config.label_loss)
        criterion_reconstruct = get_loss_function_with_logit(config.rec_loss)

        # labels
        # real_label = torch.unsqueeze(torch.ones(batch_size),1).float()
        # fake_label = torch.unsqueeze(torch.zeros(batch_size),1).float()

        if optimizer_idx == 0:
            ############
            #   VAE
            ############

            reconstructed_data, mu, logVar = vae(dataset_batch, output_all=True)
            # add instance noise
            reconstructed_data = self.add_noise_to_samples(
                reconstructed_data, instance_noise
            )

            vae_rec_loss = criterion_reconstruct(reconstructed_data, dataset_batch)

            # add KL loss
            KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1.0 - logVar)
            vae_rec_loss += KL

            # output of the vae should fool discriminator
            vae_out_d = discriminator(F.tanh(reconstructed_data))
            vae_d_loss = -vae_out_d.mean()  # vae/generator should maximize vae_out_d

            vae_loss = (vae_rec_loss + vae_d_loss) / 2

            # record loss
            self.vae_ep_loss.append(vae_loss.detach().cpu().numpy())

            return vae_loss

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############

            # generate fake trees
            latent_size = vae.decoder_z_size
            z = (
                torch.randn(batch_size, latent_size).float().type_as(dataset_batch)
            )  # noise vector
            tree_fake = F.tanh(vae.generate_sample(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            tree_fake = tree_fake.clone().detach()

            # fake data (data from generator)
            dout_fake = discriminator(tree_fake)  # detach so no update to generator
            # dloss_fake = criterion_label(dout_fake, fake_label)
            # real data (data from dataloader)
            dout_real = discriminator(dataset_batch)
            # dloss_real = criterion_label(dout_real, real_label)

            # dloss = (dloss_fake + dloss_real) / 2   #scale the loss to one
            # add gradient penalty
            gp = self.gradient_penalty(dataset_batch, tree_fake)

            dloss = (
                dout_fake.mean() - dout_real.mean()
            ) + gp  # d should maximize diff of real vs fake (dout_real - dout_fake)

            # record loss
            self.d_ep_loss.append(dloss.detach().cpu().numpy())

            return dloss

    def configure_optimizers(self):
        config = self.config
        vae = self.vae
        discriminator = self.discriminator

        self.vae_optimizer = optim.Adam(vae.parameters(), lr=config.vae_lr)

        self.discriminator_optimizer = optim.Adam(
            discriminator.parameters(), lr=config.d_lr
        )

        # clip critic (discriminator) gradient
        # no clip when gp is applied

        # clip_value = config.clip_value
        # for p in discriminator.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return self.vae_optimizer, self.discriminator_optimizer


class GAN_Wloss(GAN):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # important argument
        parser.add_argument("--clip_value", type=float, default=0.01)

        return GAN.add_model_specific_args(parser, resolution)

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0]
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add instance noise
        instance_noise = self.generate_noise_for_samples(dataset_batch)
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch, instance_noise)

        dataset_batch = dataset_batch.float()

        batch_size = dataset_batch.shape[0]
        generator = self.generator
        discriminator = self.discriminator

        # loss function
        criterion_label = get_loss_function_with_logit(config.label_loss)

        # label not used in Wloss
        if optimizer_idx == 0:
            ############
            #   generator
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # tree_fake is already computed above
            dout_fake = self.discriminator(tree_fake, output_all=False)

            # generator should maximize dout_fake
            gloss = -dout_fake.mean()

            # record loss
            self.g_ep_loss.append(gloss.detach().cpu().numpy())

            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch, output_all=False)

            # fake data (data from generator)
            dout_fake = self.discriminator(
                tree_fake.clone().detach()
            )  # detach so no update to generator

            # d should maximize diff of real vs fake (dout_real - dout_fake)
            dloss = dout_fake.mean() - dout_real.mean()

            # record loss
            self.d_ep_loss.append(dloss.detach().cpu().numpy())

            return dloss

    def configure_optimizers(self):
        config = self.config
        generator = self.generator
        discriminator = self.discriminator

        # optimizer
        self.generator_optimizer = get_model_optimizer(
            generator, config.gen_opt, config.g_lr
        )
        self.discriminator_optimizer = get_model_optimizer(
            discriminator, config.dis_opt, config.d_lr
        )

        # clip critic (discriminator) gradient
        # no clip when gp is applied

        clip_value = config.clip_value
        for p in discriminator.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        if hasattr(config, "cyclicLR_magnitude"):
            g_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.generator_optimizer,
                base_lr=config.g_lr / config.cyclicLR_magnitude,
                max_lr=config.g_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            d_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.discriminator_optimizer,
                base_lr=config.d_lr / config.cyclicLR_magnitude,
                max_lr=config.d_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            return [self.generator_optimizer, self.discriminator_optimizer], [
                g_scheduler,
                d_scheduler,
            ]
        else:
            return self.generator_optimizer, self.discriminator_optimizer


class GAN_Wloss_GP(GAN):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # important argument
        parser.add_argument("--gp_epsilon", type=float, default=2.0)

        return GAN.add_model_specific_args(parser, resolution)

    def gradient_penalty(self, real_tree, generated_tree):
        batch_size = real_tree.shape[0]

        # Calculate interpolation
        alpha = (
            torch.rand(batch_size)
            .reshape((batch_size, 1, 1, 1, 1))
            .float()
            .type_as(real_tree)
        )
        # alpha = alpha.expand_as(real_data)
        # if self.use_cuda:
        #     alpha = alpha.cuda()
        interpolated = alpha * real_tree + (1 - alpha) * generated_tree
        interpolated = interpolated.requires_grad_().float()

        # calculate prob of interpolated trees
        prob_interpolated = self.discriminator(interpolated)

        # grad tensor (all 1 to backprop some values)
        grad_tensor = torch.ones(prob_interpolated.size()).float().type_as(real_tree)

        # calculate grad of prob
        grad = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=grad_tensor,
            create_graph=True,
            retain_graph=True,
        )[0]

        # grad have shape same as input (batch_size,1,h,w,d),
        grad = grad.view(batch_size, -1)

        # calculate norm and add epsilon to prevent sqrt(0)
        grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1e-10)

        # return gradient penalty
        return self.config.gp_epsilon * ((grad_norm - 1) ** 2).mean()

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0]
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add instance noise
        instance_noise = self.generate_noise_for_samples(dataset_batch)
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch, instance_noise)

        dataset_batch = dataset_batch.float()

        batch_size = dataset_batch.shape[0]
        generator = self.generator
        discriminator = self.discriminator

        # loss function
        criterion_label = get_loss_function_with_logit(config.label_loss)

        # label not used in Wloss
        if optimizer_idx == 0:
            ############
            #   generator
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # tree_fake is already computed above
            dout_fake = self.discriminator(tree_fake, output_all=False)

            # generator should maximize dout_fake
            gloss = -dout_fake.mean()

            # record loss
            self.g_ep_loss.append(gloss.detach().cpu().numpy())

            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch, output_all=False)

            # fake data (data from generator)
            dout_fake = self.discriminator(
                tree_fake.clone().detach()
            )  # detach so no update to generator

            gp = self.gradient_penalty(dataset_batch, tree_fake)
            # d should maximize diff of real vs fake (dout_real - dout_fake)
            dloss = dout_fake.mean() - dout_real.mean() + gp

            # record loss
            self.d_ep_loss.append(dloss.detach().cpu().numpy())

            return dloss


#####
#   conditional models for training
#####
class CGAN(GAN):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # log argument
        parser.add_argument("--num_classes", type=int, default=10)
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--class_loss", type=str, default="CrossEntropyLoss")

        return GAN.add_model_specific_args(parser, resolution)

    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config
        self.save_hyperparameters("config")
        # add missing default parameters
        parser = self.add_model_specific_args(
            ArgumentParser(), resolution=config.resolution
        )
        args = parser.parse_args([])
        config = combine_namespace(args, config)
        self.config = config

        # create components
        generator = Generator(
            config.g_layer,
            config.z_size + config.num_classes,
            config.resolution,
            config.gen_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        discriminator = Discriminator(
            config.d_layer,
            config.z_size,
            config.resolution,
            config.dis_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        classifier = Discriminator(
            config.d_layer,
            config.z_size,
            config.resolution,
            config.dis_num_layer_unit,
            output_size=config.num_classes,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        # GAN
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier

        # others
        self.noise_magnitude = config.instance_noise

    def forward(self, x):
        # classifier and discriminator
        x = self.generator(x)
        x = F.tanh(x)
        x = self.discriminator(x)
        c = self.classifier(x)
        return x, c

    def configure_optimizers(self):
        config = self.config
        generator = self.generator
        discriminator = self.discriminator
        classifier = self.classifier

        # optimizer
        self.generator_optimizer = get_model_optimizer(
            generator, config.gen_opt, config.g_lr
        )
        self.discriminator_optimizer = get_model_optimizer(
            discriminator, config.dis_opt, config.d_lr
        )
        self.classifier_optimizer = get_model_optimizer(
            classifier, config.dis_opt, config.d_lr
        )

        if hasattr(config, "cyclicLR_magnitude"):
            g_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.generator_optimizer,
                base_lr=config.g_lr / config.cyclicLR_magnitude,
                max_lr=config.g_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            d_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.discriminator_optimizer,
                base_lr=config.d_lr / config.cyclicLR_magnitude,
                max_lr=config.d_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            c_scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.classifier_optimizer,
                base_lr=config.d_lr / config.cyclicLR_magnitude,
                max_lr=config.d_lr * config.cyclicLR_magnitude,
                step_size_up=200,
            )
            return [
                self.generator_optimizer,
                self.discriminator_optimizer,
                self.classifier_optimizer,
            ], [g_scheduler, d_scheduler, c_scheduler]
        else:
            return (
                self.generator_optimizer,
                self.discriminator_optimizer,
                self.classifier_optimizer,
            )

    def on_train_epoch_start(self):
        # reset ep_loss
        self.d_ep_loss = []
        self.g_ep_loss = []
        self.c_ep_loss = []

        # calc instance noise
        if self.noise_magnitude > 0:
            # linear annealed noise
            noise_rate = (
                self.config.linear_annealed_instance_noise_epoch - self.current_epoch
            ) / self.config.linear_annealed_instance_noise_epoch
            self.noise_magnitude = self.config.instance_noise * noise_rate
        else:
            self.noise_magnitude = 0

        # set model to train
        self.generator.train()
        self.discriminator.train()
        self.classifier.train()

    def on_train_epoch_end(self, epoch_output):

        # save model if necessary
        log_dict = {
            "classifier loss": np.mean(self.c_ep_loss),
            "discriminator loss": np.mean(self.d_ep_loss),
            "generator loss": np.mean(self.g_ep_loss),
            "epoch": self.current_epoch,
        }

        log_media = (
            self.current_epoch % self.config.log_interval == 0
        )  # boolean whether to log image/3D object

        wandbLog_cond(
            self,
            self.config.num_classes,
            log_dict,
            log_media=log_media,
            log_num_samples=self.config.log_num_samples,
        )

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        dataset_batch, dataset_indices = dataset_batch
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add instance noise
        instance_noise = self.generate_noise_for_samples(dataset_batch)
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch, instance_noise)

        dataset_batch = dataset_batch.float()
        dataset_indices = dataset_indices.to(torch.int64)

        batch_size = dataset_batch.shape[0]
        generator = self.generator
        discriminator = self.discriminator
        classifier = self.classifier

        # loss function
        criterion_label = get_loss_function_with_logit(config.label_loss)
        criterion_class = get_loss_function_with_logit(config.class_loss)

        # labels
        # soft label
        # modified scale to [1-label_noise,1]
        # modified scale to [0,label_noise]
        real_label = 1 - (torch.rand(batch_size) * config.label_noise)
        fake_label = torch.rand(batch_size) * config.label_noise
        # add noise to label
        # P(label_flip_prob) label flipped
        label_flip_mask = torch.bernoulli(
            torch.ones(batch_size) * config.label_flip_prob
        )
        real_label = (1 - label_flip_mask) * real_label + label_flip_mask * (
            1 - real_label
        )
        fake_label = (1 - label_flip_mask) * fake_label + label_flip_mask * (
            1 - fake_label
        )

        real_label = torch.unsqueeze(real_label, 1).float().type_as(dataset_batch)
        fake_label = torch.unsqueeze(fake_label, 1).float().type_as(dataset_batch)
        if optimizer_idx == 0:
            ############
            #   generator
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            c_fake = (
                torch.randint(0, config.num_classes, (batch_size,))
                .type_as(dataset_batch)
                .to(torch.int64)
            )  # class vector

            # convert c to one-hot
            batch_size = z.shape[0]
            c = c_fake.reshape((-1, 1))
            c_onehot = torch.zeros([batch_size, config.num_classes]).type_as(
                dataset_batch
            )
            c_onehot = c_onehot.scatter(1, c, 1)

            # merge with z to be generator input
            z = torch.cat((z, c_onehot), 1)

            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # tree_fake on Dis
            dout_fake = self.discriminator(tree_fake, output_all=False)
            # generator should generate trees that discriminator think they are real
            gloss_d = criterion_label(dout_fake, real_label)

            # tree_fake on Cla
            cout_fake = self.classifier(tree_fake, output_all=False)
            gloss_c = criterion_class(cout_fake, c_fake)

            gloss = (gloss_d + gloss_c) / 2
            # record loss
            self.g_ep_loss.append(gloss.detach().cpu().numpy())

            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            c = (
                torch.randint(0, config.num_classes, (batch_size,))
                .type_as(dataset_batch)
                .to(torch.int64)
            )  # class vector

            # convert c to one-hot
            batch_size = z.shape[0]
            c = c.reshape((-1, 1))
            c_onehot = torch.zeros([batch_size, config.num_classes]).type_as(
                dataset_batch
            )
            c_onehot = c_onehot.scatter(1, c, 1)

            # merge with z to be decoder input
            z = torch.cat((z, c_onehot), 1)

            # detach so no update to generator
            tree_fake = F.tanh(self.generator(z)).clone().detach()
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch, output_all=False)
            dloss_real = criterion_label(dout_real, real_label)

            # fake data (data from generator)
            dout_fake = self.discriminator(tree_fake)
            dloss_fake = criterion_label(dout_fake, fake_label)

            # loss function (discriminator classify real data vs generated data)
            dloss = (dloss_real + dloss_fake) / 2

            # record loss
            self.d_ep_loss.append(dloss.detach().cpu().numpy())

            return dloss

        if optimizer_idx == 2:

            ############
            #   classifier
            ############

            z = (
                torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            )  # 128-d noise vector
            c_fake = (
                torch.randint(0, config.num_classes, (batch_size,))
                .type_as(dataset_batch)
                .to(torch.int64)
            )  # class vector

            # convert c to one-hot
            batch_size = z.shape[0]
            c = c_fake.reshape((-1, 1))
            c_onehot = torch.zeros([batch_size, config.num_classes]).type_as(
                dataset_batch
            )
            c_onehot = c_onehot.scatter(1, c, 1)

            # merge with z to be generator input
            z = torch.cat((z, c_onehot), 1)

            # detach so no update to generator
            tree_fake = F.tanh(self.generator(z)).clone().detach()
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake, instance_noise)

            # fake data (data from generator)
            cout_fake = self.classifier(tree_fake)
            closs_fake = criterion_class(cout_fake, c_fake)

            # real data (data from dataloader)
            cout_real = self.classifier(dataset_batch, output_all=False)
            closs_real = criterion_class(cout_real, dataset_indices)

            # loss function (discriminator classify real data vs generated data)
            closs = (closs_real + closs_fake) / 2

            # record loss
            self.c_ep_loss.append(closs.detach().cpu().numpy())

            return closs

    def generate_tree(self, c, num_trees=1):
        config = self.config
        generator = self.generator

        return generate_tree(
            generator,
            config.resolution,
            c,
            config.num_classes,
            num_trees=num_trees,
            batch_size=config.batch_size,
        )


#####
#   building blocks for models
#####


class Generator(nn.Module):
    def __init__(
        self,
        layer_per_block=2,
        z_size=128,
        output_size=64,
        num_layer_unit=32,
        dropout_prob=0.3,
        spectral_norm=False,
        activations=nn.ReLU(True),
    ):
        super(Generator, self).__init__()
        self.z_size = z_size

        # layer_per_block must be >= 1
        if layer_per_block < 1:
            layer_per_block = 1

        self.fc_channel = 8  # 16
        self.fc_size = 4

        self.output_size = output_size
        # need int(output_size / self.fc_size) upsampling to increase size, so we have int(output_size / self.fc_size) + 1 block
        self.num_blocks = int(np.log2(output_size) - np.log2(self.fc_size)) + 1

        if type(num_layer_unit) is list:
            if len(num_layer_unit) != self.num_blocks:
                raise Exception(
                    "For output_size="
                    + str(output_size)
                    + ", the list of num_layer_unit should have "
                    + str(self.num_blocks)
                    + " elements."
                )
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
            num_layer_unit1, num_layer_unit2 = (
                num_layer_unit_list[i],
                num_layer_unit_list[i + 1],
            )

            for _ in range(layer_per_block):
                if spectral_norm:
                    gen_module.append(
                        SpectralNorm(
                            nn.ConvTranspose3d(
                                num_layer_unit1, num_layer_unit2, 3, 1, padding=1
                            )
                        )
                    )
                else:
                    gen_module.append(
                        nn.ConvTranspose3d(
                            num_layer_unit1, num_layer_unit2, 3, 1, padding=1
                        )
                    )
                gen_module.append(nn.BatchNorm3d(num_layer_unit2))
                gen_module.append(activations)
                gen_module.append(nn.Dropout3d(dropout_prob))
                num_layer_unit1 = num_layer_unit2

            gen_module.append(nn.Upsample(scale_factor=2, mode="trilinear"))

        # remove extra pool layer
        gen_module = gen_module[:-1]

        # remove tanh for loss with logit
        if spectral_norm:
            gen_module.append(
                SpectralNorm(
                    nn.ConvTranspose3d(num_layer_unit_list[-1], 1, 3, 1, padding=1)
                )
            )
        else:
            gen_module.append(
                nn.ConvTranspose3d(num_layer_unit_list[-1], 1, 3, 1, padding=1)
            )
        # gen_module.append(nn.tanh())

        self.gen_fc = nn.Linear(
            self.z_size, self.fc_channel * self.fc_size * self.fc_size * self.fc_size
        )
        self.gen = nn.Sequential(*gen_module)

    def forward(self, x):
        x = self.gen_fc(x)
        x = x.view(
            x.shape[0], self.fc_channel, self.fc_size, self.fc_size, self.fc_size
        )
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        layer_per_block=2,
        z_size=128,
        input_size=64,
        num_layer_unit=16,
        dropout_prob=0.3,
        output_size=1,
        spectral_norm=False,
        activations=nn.LeakyReLU(0.0, True),
    ):
        super(Discriminator, self).__init__()

        self.z_size = z_size

        # layer_per_block must be >= 1
        if layer_per_block < 1:
            layer_per_block = 1

        self.fc_size = 4  # final height of the volume in conv layers before flatten

        self.input_size = input_size
        self.output_size = output_size
        # need int(input_size / self.fc_size) upsampling to increase size, so we have int(input_size / self.fc_size) + 1 block
        self.num_blocks = int(np.log2(input_size) - np.log2(self.fc_size)) + 1

        if type(num_layer_unit) is list:
            if len(num_layer_unit) != self.num_blocks:
                raise Exception(
                    "For input_size="
                    + str(input_size)
                    + ", the list of num_layer_unit should have "
                    + str(self.num_blocks)
                    + " elements."
                )
            num_layer_unit_list = num_layer_unit
        elif type(num_layer_unit) is int:
            num_layer_unit_list = [num_layer_unit] * self.num_blocks
        else:
            raise Exception("num_layer_unit should be int of list of int.")

        # add initial num_unit to num_layer_unit_list
        num_layer_unit_list = [1] + num_layer_unit_list
        dis_module = []
        # 5 blocks (need 4 pool to reduce size)
        for i in range(self.num_blocks):
            num_layer_unit1, num_layer_unit2 = (
                num_layer_unit_list[i],
                num_layer_unit_list[i + 1],
            )

            for _ in range(layer_per_block):
                if spectral_norm:
                    dis_module.append(
                        SpectralNorm(
                            nn.Conv3d(num_layer_unit1, num_layer_unit2, 3, 1, padding=1)
                        )
                    )
                else:
                    dis_module.append(
                        nn.Conv3d(num_layer_unit1, num_layer_unit2, 3, 1, padding=1)
                    )
                dis_module.append(nn.BatchNorm3d(num_layer_unit2))
                dis_module.append(activations)
                dis_module.append(nn.Dropout3d(dropout_prob))
                num_layer_unit1 = num_layer_unit2

            dis_module.append(nn.MaxPool3d((2, 2, 2)))

        # remove extra pool layer
        dis_module = dis_module[:-1]

        self.dis = nn.Sequential(*dis_module)

        self.dis_fc1 = nn.Sequential(
            nn.Linear(
                num_layer_unit_list[-1] * self.fc_size * self.fc_size * self.fc_size,
                z_size,
            ),
            nn.ReLU(True),
        )
        self.dis_fc2 = nn.Sequential(
            nn.Linear(z_size, self.output_size),
            # nn.tanh()  #remove tanh for loss with logit
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
        assert encoder.input_size == decoder.output_size
        self.sample_size = decoder.output_size
        self.encoder_z_size = encoder.z_size
        self.decoder_z_size = decoder.z_size
        # VAE
        self.vae_encoder = encoder
        self.encoder_mean = nn.Linear(self.encoder_z_size, self.decoder_z_size)
        self.encoder_logvar = nn.Linear(self.encoder_z_size, self.decoder_z_size)
        self.vae_decoder = decoder

    # reference: https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner/blob/master/CVAE-GAN/CVAE-GAN.py
    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).type_as(mean)
        z = mean + eps * torch.exp(logvar / 2.0)
        return z

    def forward(self, x, output_all=False):
        # VAE
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


class CVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE, self).__init__()
        assert encoder.input_size == decoder.output_size
        self.sample_size = decoder.output_size
        self.encoder_z_size = encoder.z_size
        self.decoder_z_size = decoder.z_size
        self.num_classes = self.decoder_z_size - self.encoder_z_size
        # CVAE
        self.vae_encoder = encoder
        self.encoder_mean = nn.Linear(self.encoder_z_size, self.encoder_z_size)
        self.encoder_logvar = nn.Linear(self.encoder_z_size, self.encoder_z_size)
        self.vae_decoder = decoder

    # reference: https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner/blob/master/CVAE-GAN/CVAE-GAN.py
    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).type_as(mean)
        z = mean + eps * torch.exp(logvar / 2.0)
        return z

    def forward(self, x, c, output_all=False):
        # CVAE
        _, f = self.vae_encoder(x, output_all=True)
        x_mean = self.encoder_mean(f)
        x_logvar = self.encoder_logvar(f)
        x = self.noise_reparameterize(x_mean, x_logvar)

        # convert c to one-hot
        batch_size = x.shape[0]
        c = c.reshape((-1, 1))
        c_onehot = torch.zeros([batch_size, self.num_classes])
        c_onehot = c_onehot.scatter(1, c, 1)

        # merge with x to be decoder input
        x = torch.cat((x, c_onehot), 1)

        x = self.vae_decoder(x)
        if output_all:
            return x, x_mean, x_logvar
        else:
            return x

    def generate_sample(self, z, c):
        # convert c to one-hot
        batch_size = z.shape[0]
        c = c.reshape((-1, 1))
        c_onehot = torch.zeros([batch_size, self.num_classes])
        c_onehot = c_onehot.scatter(1, c, 1)

        # merge with z to be decoder input
        z = torch.cat((z, c_onehot), 1)

        x = self.vae_decoder(z)
        return x


###
#       other modules
###

#
#   reference: https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
#
class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data)
            )
            u.data = self.l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


###
#       functions
###


def generate_tree(
    generator, resolution, c=None, num_classes=None, num_trees=1, batch_size=-1
):
    if batch_size == -1:
        batch_size = 32

    if batch_size > num_trees:
        batch_size = num_trees

    result = None

    num_runs = int(np.ceil(num_trees / batch_size))
    # ignore discriminator
    for i in range(num_runs):

        if c is not None:
            # generate noise vector
            z = torch.randn(batch_size, generator.z_size - num_classes).type_as(
                generator.gen_fc.weight
            )
            # convert c to one-hot
            batch_size = z.shape[0]
            c_onehot = torch.zeros([batch_size, num_classes]).type_as(z)
            c_onehot[:, c] = 1
            # merge with z to be generator input
            z = torch.cat((z, c_onehot), 1)
        else:
            # generate noise vector
            z = torch.randn(batch_size, generator.z_size).type_as(
                generator.gen_fc.weight
            )

        # no tanh so hasvoxel means >0
        tree_fake = generator(z)[:, 0, :, :, :]
        selected_trees = tree_fake.detach().cpu().numpy()
        if result is None:
            result = selected_trees
        else:
            result = np.concatenate((result, selected_trees), axis=0)

    # select at most num_trees
    if result.shape[0] > num_trees:
        result = result[:num_trees]
    # in case no good result
    if result.shape[0] <= 0:
        result = np.zeros((1, resolution, resolution, resolution))
        result[:, 0, 0, 0] = 1
    return result


def get_model_optimizer(model, optimizer_option, lr):

    if optimizer_option == "Adam":
        optimizer = optim.Adam
    elif optimizer_option == "SGD":
        optimizer = optim.SGD
    else:
        raise Exception(
            "optimizer_option must be in ['Adam', 'SGD']. Current "
            + str(optimizer_option)
        )
    return optimizer(model.parameters(), lr=lr)


def get_loss_function_with_logit(loss_option):

    if loss_option == "BCELoss":
        loss = nn.BCEWithLogitsLoss(reduction="mean")
    elif loss_option == "MSELoss":
        loss = lambda gen, data: nn.MSELoss(reduction="mean")(F.tanh(gen), data)
    elif loss_option == "CrossEntropyLoss":
        loss = nn.CrossEntropyLoss(reduction="mean")
    else:
        raise Exception(
            "loss_option must be in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']. Current "
            + str(loss_option)
        )
    return loss


def combine_namespace(base, update):
    base = vars(base)
    base.update(vars(update))
    return Namespace(**base)
