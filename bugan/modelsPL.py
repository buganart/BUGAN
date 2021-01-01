from bugan.functionsPL import *
from bugan.datamodulePL import DataModule_process, DataModule_process_cond


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
#   Base model
#   The parent of all training models
#
#   This model manages settings that shares among all training models, including
#   1) setup model components (optimizer, setup on_epoch / on_batch, logging, ...)
#   2) wandb logging (log when epoch end)
#   3) GAN hacks (label_noise, instance_noise, dropout_prob, spectral_norm, ...)
#   4) other common functions (generate_trees, get_loss_function_with_logit, create_real_fake_label, ...)
#   5) LightningModule functions (configure_optimizers, on_train_epoch_start, on_train_epoch_end)
#   *) Note that __init__() and training_step should be implemented in child model
#####
class BaseModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # basic attributes for training
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--resolution", type=int, default=resolution)
        # wandb log argument
        parser.add_argument("--log_interval", type=int, default=10)
        parser.add_argument("--log_num_samples", type=int, default=1)

        # label noise
        # real/fake label flip probability
        parser.add_argument("--label_flip_prob", type=float, default=0.1)
        # real/fake label noise magnitude
        parser.add_argument("--label_noise", type=float, default=0.2)

        # instance noise (add noise to real data/generate data)
        # generate instance noise once per batch?
        # if False, generate instance noise for every sample,
        # including every generated data and real data
        # if True, same noise will be applied to
        # every generated data and real data in the same batch
        parser.add_argument("--instance_noise_per_batch", type=bool, default=True)
        # noise linear delay time
        parser.add_argument(
            "--linear_annealed_instance_noise_epoch", type=int, default=1000
        )
        parser.add_argument("--instance_noise", type=float, default=0.3)

        # default Generator/Discriminator parameters
        # latent vector size
        parser.add_argument("--z_size", type=int, default=128)
        # activation default leakyReLU
        parser.add_argument("--activation_leakyReLU_slope", type=float, default=0.1)
        # Dropout probability
        parser.add_argument("--dropout_prob", type=float, default=0.3)
        # spectral_norm
        parser.add_argument("--spectral_norm", type=bool, default=False)
        # use_simple_3dgan_struct
        # reference: https://github.com/xchhuang/simple-pytorch-3dgan
        # basically both G and D will conv the input to volume (1,1,1),
        # which is different from the other one that flatten when (k,k,k), k=4
        parser.add_argument("--use_simple_3dgan_struct", type=bool, default=False)

        return parser

    #####
    #   __init__() related functions
    #####

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = self.setup_config_arguments(config)
        # list for setup_model_component()
        # record (model_list, opt_config_list) for configure_optimizers()
        # record (model_name_list, model_ep_loss_list) for logging loss
        # * the items for the same model component should have the same list index
        # * the list index also indicates the optimizer_idx in the training_step

        # a list containing all model components
        self.model_list = []
        # model name string is recorded only for logging purpose
        self.model_name_list = []
        # config for setup optimizer for the components in self.model_list
        self.opt_config_list = []
        # record loss of each model component in training_step
        self.model_ep_loss_list = []

        # for instance noise
        self.noise_magnitude = self.config.instance_noise
        self.instance_noise = None

    # The add_model_specific_args() function listed all the parameters for the model
    # from the config dict given by the user, the necessary parameters may not be there
    # This function takes all the default values from the add_model_specific_args() function,
    # and then add only the missing parameter values to the user config dict
    # (the values in add_model_specific_args() will be overwritten by values in user config dict)
    # * this function should be in every child model __init__() right after super().__init__()
    def setup_config_arguments(self, config):
        if hasattr(config, "cyclicLR_magnitude"):
            resolution = config.resolution
        else:
            resolution = 32
        # add missing default parameters
        parser = self.add_model_specific_args(ArgumentParser(), resolution=resolution)
        args = parser.parse_args([])
        config = self.combine_namespace(args, config)
        return config

    # helper function to combine Namespace object
    # values in base will be overwritten by values in update
    def combine_namespace(self, base, update):
        base = vars(base)
        base.update(vars(update))
        return Namespace(**base)

    def setup_Generator(
        self,
        model_name,
        layer_per_block,
        num_layer_unit,
        optimizer_option,
        learning_rate,
        num_classes=None,
    ):
        config = self.config
        if not num_classes:
            z_size = config.z_size
        else:
            z_size = config.z_size + num_classes
        generator = Generator(
            layer_per_block=layer_per_block,
            z_size=z_size,
            output_size=config.resolution,
            num_layer_unit=num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            use_simple_3dgan_struct=config.use_simple_3dgan_struct,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )
        # setup component in __init__() lists
        # for configure_optimizers() and record loss
        self.setup_model_component(
            generator, model_name, optimizer_option, learning_rate
        )
        return generator

    def setup_Discriminator(
        self,
        model_name,
        layer_per_block,
        num_layer_unit,
        optimizer_option,
        learning_rate,
        output_size=1,
    ):
        config = self.config

        discriminator = Discriminator(
            layer_per_block=layer_per_block,
            output_size=output_size,
            input_size=config.resolution,
            num_layer_unit=num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            use_simple_3dgan_struct=config.use_simple_3dgan_struct,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )
        # setup component in __init__() lists
        # for configure_optimizers() and record loss
        self.setup_model_component(
            discriminator, model_name, optimizer_option, learning_rate
        )
        return discriminator

    def setup_VAE(
        self,
        model_name,
        encoder_layer_per_block,
        encoder_num_layer_unit,
        decoder_layer_per_block,
        decoder_num_layer_unit,
        optimizer_option,
        learning_rate,
    ):
        config = self.config

        encoder = Discriminator(
            layer_per_block=encoder_layer_per_block,
            output_size=config.z_size,
            input_size=config.resolution,
            num_layer_unit=encoder_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            use_simple_3dgan_struct=config.use_simple_3dgan_struct,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        decoder = Generator(
            layer_per_block=decoder_layer_per_block,
            z_size=config.z_size,
            output_size=config.resolution,
            num_layer_unit=decoder_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            use_simple_3dgan_struct=config.use_simple_3dgan_struct,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        vae = VAE(encoder=encoder, decoder=decoder)
        # setup component in __init__() lists
        # for configure_optimizers() and record loss
        self.setup_model_component(vae, model_name, optimizer_option, learning_rate)
        return vae

    # setup model components to the lists for later use
    # record (model_list, opt_config_list) for configure_optimizers()
    # record (model_name_list, model_ep_loss_list) for logging loss
    # * this function will be called when initializing model components
    #   in setup_Generator/Discriminator/VAE
    def setup_model_component(self, model, model_name, model_opt_string, model_lr):
        self.model_list.append(model)
        self.model_name_list.append(model_name)
        self.opt_config_list.append((model_opt_string, model_lr))
        self.model_ep_loss_list.append([])

    # return torch loss function based on the string loss_option
    # the returned loss assume input to be logit (before sigmoid/tanh)
    def get_loss_function_with_logit(self, loss_option):

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

    #####
    #   configure_optimizers()
    #####
    def configure_optimizers(self):
        if hasattr(self.config, "cyclicLR_magnitude"):
            cyclicLR_magnitude = self.config.cyclicLR_magnitude
        else:
            cyclicLR_magnitude = None

        optimizer_list = []
        scheduler_list = []

        # setup optimizer for each model component in model_list
        # (check __init__() and setup_model_component())
        # if cyclicLR_magnitude is set in user config, also setup scheduler
        for idx in range(len(self.model_list)):
            model = self.model_list[idx]
            model_opt_string, model_lr = self.opt_config_list[idx]
            optimizer = self.get_model_optimizer(model, model_opt_string, model_lr)
            optimizer_list.append(optimizer)
            # check if use cyclicLR
            if cyclicLR_magnitude:
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=model_lr / cyclicLR_magnitude,
                    max_lr=model_lr * cyclicLR_magnitude,
                    step_size_up=200,
                )
                scheduler_list.append(scheduler)

        if cyclicLR_magnitude:
            return optimizer_list, scheduler_list
        else:
            return optimizer_list

    # return torch optimizer based on the string optimizer_option
    def get_model_optimizer(self, model, optimizer_option, lr):

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

    #####
    #   on_train_epoch_start()
    #####
    def on_train_epoch_start(self):
        # reset ep_loss
        # set model to train
        for idx in range(len(self.model_ep_loss_list)):
            self.model_ep_loss_list[idx] = []
            self.model_list[idx].train()

        # calc instance noise
        # check add_noise_to_samples() and generate_noise_for_samples()
        if self.noise_magnitude > 0:
            # linear annealed noise
            noise_rate = (
                self.config.linear_annealed_instance_noise_epoch - self.current_epoch
            ) / self.config.linear_annealed_instance_noise_epoch
            self.noise_magnitude = self.config.instance_noise * noise_rate
        else:
            self.noise_magnitude = 0

    #####
    #   on_train_batch_start()
    #####
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # record/reset instance noise per batch
        self.instance_noise = None

    #####
    #   on_train_epoch_end()
    #####
    def on_train_epoch_end(self, epoch_output):
        log_dict = {"epoch": self.current_epoch}

        # record loss and add to log_dict
        for idx in range(len(self.model_ep_loss_list)):
            loss = np.mean(self.model_ep_loss_list[idx])
            loss_name = self.model_name_list[idx] + " loss"
            log_dict[loss_name] = loss

        # boolean whether to log image/3D object
        log_media = self.current_epoch % self.config.log_interval == 0

        # log data to wandb
        if hasattr(self, "classifier"):
            # conditional
            class_list = self.trainer.datamodule.class_list
        else:
            # unconditional
            class_list = None

        self.wandbLog(
            class_list,
            log_dict,
            log_media=log_media,
            log_num_samples=self.config.log_num_samples,
        )

    # construct log_dict to log data/statistics to wandb
    # image/mesh/statistics are calculated by calculate_log_media_stat() in functionPL.py
    # mesh data statistics
    # 1) average number of voxel per tree
    # 2) average number of voxel cluster per tree (check distance function)
    # 3) images of all generated tree
    # 4) meshes of all generated tree
    # 5) mean of per voxel std over generated trees
    def wandbLog(
        self, class_list=None, initial_log_dict={}, log_media=False, log_num_samples=1
    ):

        if log_media:

            if class_list is not None:
                num_classes = len(class_list)
                # log condition model data
                for c in range(num_classes):
                    (
                        numpoints,
                        num_cluster,
                        image,
                        voxelmesh,
                        std,
                    ) = calculate_log_media_stat(self, log_num_samples, class_label=c)

                    # add list record to log_dict
                    initial_log_dict[
                        "sample_tree_numpoints_class_"
                        + str(c)
                        + "_"
                        + str(class_list[c])
                    ] = numpoints
                    initial_log_dict[
                        "eval_num_cluster_class_" + str(c) + "_" + str(class_list[c])
                    ] = num_cluster
                    initial_log_dict[
                        "sample_tree_image_class_" + str(c) + "_" + str(class_list[c])
                    ] = image
                    initial_log_dict[
                        "sample_tree_voxelmesh_class_"
                        + str(c)
                        + "_"
                        + str(class_list[c])
                    ] = voxelmesh
                    initial_log_dict[
                        "mesh_per_voxel_std_class_" + str(c) + "_" + str(class_list[c])
                    ] = std
            else:
                # log uncondition model data
                (
                    numpoints,
                    num_cluster,
                    image,
                    voxelmesh,
                    std,
                ) = calculate_log_media_stat(self, log_num_samples)

                # add list record to log_dict
                initial_log_dict["sample_tree_numpoints"] = numpoints
                initial_log_dict["eval_num_cluster"] = num_cluster
                initial_log_dict["sample_tree_image"] = image
                initial_log_dict["sample_tree_voxelmesh"] = voxelmesh
                initial_log_dict["mesh_per_voxel_std"] = std

        wandb.log(initial_log_dict)

    #####
    #   training_step() related function
    #####
    def training_step(self, dataset_batch, batch_idx):
        # implement training_step() in child model
        pass

    # create true/false label for discriminator loss
    # normally, true label is 1 and false label is 0
    # this function also add noise to the labels,
    # so true:[1-label_noise,1], false:[0,label_noise]
    # also, flip label with P(config.label_flip_prob)
    def create_real_fake_label(self, dataset_batch):
        config = self.config
        batch_size = dataset_batch.shape[0]
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
        return real_label, fake_label

    # calculate KL loss for VAE
    def calculate_KL_loss(self, mu, logVar):
        KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1.0 - logVar)
        return KL

    # save loss to list for updating loss on wandb log
    def record_loss(self, loss, optimizer_idx):
        self.model_ep_loss_list[optimizer_idx].append(loss)

    # accuracy hack
    # stop update discriminator when prediction accurary > config.accuracy_hack
    # stop update by return 0 loss (dloss - dloss)
    # *** the parameter config.accuracy_hack is not in
    #     BaseModel.add_model_specific_args().
    #     Please add accuracy_hack in the child class.
    def apply_accuracy_hack(self, dloss, dout_real, dout_fake):
        config = self.config
        # accuracy hack
        if config.accuracy_hack < 1.0:
            # hack activated, calculate accuracy
            # note that dout are before sigmoid
            real_score = (dout_real >= 0).float()
            fake_score = (dout_fake < 0).float()
            accuracy = torch.cat((real_score, fake_score), 0).mean()
            if accuracy > config.accuracy_hack:
                return dloss - dloss
        return dloss

    # for conditional models
    # given latent_vector (B, Z) and class_vector (B),
    # reshape class_vector to one-hot (B, num_classes),
    # and merge with latent_vector
    def merge_latent_and_class_vector(self, latent_vector, class_vector):
        z = latent_vector
        c = class_vector
        batch_size = z.shape[0]
        # convert c to one-hot
        c = c.reshape((-1, 1))
        c_onehot = torch.zeros([batch_size, self.config.num_classes]).type_as(z)
        c_onehot = c_onehot.scatter(1, c, 1)

        # merge with z to be generator input
        z = torch.cat((z, c_onehot), 1)
        return z

    # instance noise
    # create and add instance noise which has the same shape as the data
    def add_noise_to_samples(self, data):
        if self.noise_magnitude <= 0:
            return data

        # self.instance_noise will always be None if
        # self.config.instance_noise_per_batch is False
        if self.instance_noise is not None:
            noise = self.instance_noise
        else:
            # create uniform noise
            noise = torch.rand(data.shape) * 2 - 1
            noise = self.noise_magnitude * noise  # noise in [-magn, magn]
            noise = noise.float().type_as(data).detach()

        # share the same noise for real and generated data in the same batch
        if self.config.instance_noise_per_batch:
            self.instance_noise = noise
        # add instance noise
        # now batch in [-1+magn, 1-magn]
        data = data * (1 - self.noise_magnitude)
        data = data + noise
        return data

    # generate tree
    # for unconditional model, this takes the generator of the model and generate trees
    # for conditional model, this also take class label c and num_classes,
    # to generate trees of the specified class c
    # this function will generate n trees per call (n = num_trees)
    def generate_tree(self, generator, c=None, num_classes=None, num_trees=1):
        batch_size = self.config.batch_size
        resolution = generator.output_size

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


#####
#   models for training
#####
class VAE_train(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # model specific argument (VAE)

        # number of layer per block
        parser.add_argument("--vae_decoder_layer", type=int, default=1)
        parser.add_argument("--vae_encoder_layer", type=int, default=1)
        # optimizer in {"Adam", "SGD"}
        parser.add_argument("--vae_opt", type=str, default="Adam")
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--rec_loss", type=str, default="MSELoss")
        # learning rate
        parser.add_argument("--vae_lr", type=float, default=0.0005)
        # number of unit per layer
        if resolution == 32:
            decoder_num_layer_unit = [1024, 512, 256, 128]
            encoder_num_layer_unit = [32, 64, 128, 128]
        else:
            decoder_num_layer_unit = [1024, 512, 256, 128, 128]
            encoder_num_layer_unit = [32, 64, 64, 128, 128]
        parser.add_argument("--decoder_num_layer_unit", default=decoder_num_layer_unit)
        parser.add_argument("--encoder_num_layer_unit", default=encoder_num_layer_unit)

        return BaseModel.add_model_specific_args(parser)

    def __init__(self, config):
        super(VAE_train, self).__init__(config)
        # assert(vae.sample_size == discriminator.input_size)
        self.config = config
        self.save_hyperparameters("config")
        config = self.setup_config_arguments(config)
        self.config = config

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type
        self.vae = self.setup_VAE(
            model_name="VAE",
            encoder_layer_per_block=config.vae_encoder_layer,
            encoder_num_layer_unit=config.encoder_num_layer_unit,
            decoder_layer_per_block=config.vae_decoder_layer,
            decoder_num_layer_unit=config.decoder_num_layer_unit,
            optimizer_option=config.vae_opt,
            learning_rate=config.vae_lr,
        )

        # loss function
        self.criterion_reconstruct = self.get_loss_function_with_logit(config.rec_loss)

    def forward(self, x):
        x = self.vae(x)
        return x

    def training_step(self, dataset_batch, batch_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0].float()
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        batch_size = dataset_batch.shape[0]

        reconstructed_data, mu, logVar = self.vae(dataset_batch, output_all=True)
        # add instance noise
        reconstructed_data = self.add_noise_to_samples(reconstructed_data)

        vae_rec_loss = self.criterion_reconstruct(reconstructed_data, dataset_batch)

        # add KL loss
        KL = self.calculate_KL_loss(mu, logVar)

        vae_loss = vae_rec_loss + KL

        # record loss
        self.record_loss(vae_loss.detach().cpu().numpy(), 0)
        return vae_loss

    def generate_tree(self, num_trees=1):
        generator = self.vae.vae_decoder
        return super().generate_tree(generator, num_trees=num_trees)


class VAEGAN(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # number of layer per block
        parser.add_argument("--vae_decoder_layer", type=int, default=1)
        parser.add_argument("--vae_encoder_layer", type=int, default=1)
        parser.add_argument("--d_layer", type=int, default=1)
        # optimizer in {"Adam", "SGD"}
        parser.add_argument("--vae_opt", type=str, default="Adam")
        parser.add_argument("--dis_opt", type=str, default="Adam")
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--label_loss", type=str, default="BCELoss")
        parser.add_argument("--rec_loss", type=str, default="MSELoss")
        # accuracy_hack
        parser.add_argument("--accuracy_hack", type=float, default=1.1)
        # learning rate
        parser.add_argument("--vae_lr", type=float, default=0.0005)
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

        return BaseModel.add_model_specific_args(parser)

    def __init__(self, config):
        super(VAEGAN, self).__init__(config)
        # assert(vae.sample_size == discriminator.input_size)
        self.config = config
        self.save_hyperparameters("config")
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type
        self.vae = self.setup_VAE(
            model_name="VAE",
            encoder_layer_per_block=config.vae_encoder_layer,
            encoder_num_layer_unit=config.encoder_num_layer_unit,
            decoder_layer_per_block=config.vae_decoder_layer,
            decoder_num_layer_unit=config.decoder_num_layer_unit,
            optimizer_option=config.vae_opt,
            learning_rate=config.vae_lr,
        )
        self.discriminator = self.setup_Discriminator(
            "discriminator",
            layer_per_block=config.d_layer,
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
        )

        # loss function
        self.criterion_label = self.get_loss_function_with_logit(config.label_loss)
        self.criterion_reconstruct = self.get_loss_function_with_logit(config.rec_loss)

    def forward(self, x):
        # VAE
        x = self.vae(x)
        x = F.tanh(x)
        # classifier and discriminator
        x = self.discriminator(x)
        return x

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0].float()
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        if optimizer_idx == 0:
            ############
            #   VAE
            ############

            reconstructed_data, mu, logVar = self.vae(dataset_batch, output_all=True)
            # add noise to data
            reconstructed_data = self.add_noise_to_samples(reconstructed_data)

            vae_rec_loss = self.criterion_reconstruct(reconstructed_data, dataset_batch)

            # add KL loss
            KL = self.calculate_KL_loss(mu, logVar)
            vae_rec_loss += KL

            # output of the vae should fool discriminator
            vae_out_d = self.discriminator(F.tanh(reconstructed_data))
            vae_d_loss = self.criterion_label(vae_out_d, real_label)

            vae_loss = (vae_rec_loss + vae_d_loss) / 2

            # record loss
            self.record_loss(vae_loss.detach().cpu().numpy(), optimizer_idx)

            return vae_loss

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############

            # generate fake trees
            latent_size = self.vae.decoder_z_size
            # latent noise vector
            z = torch.randn(batch_size, latent_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.vae.generate_sample(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # fake data (data from generator)
            # detach so no update to generator
            dout_fake = self.discriminator(tree_fake.clone().detach())
            dloss_fake = self.criterion_label(dout_fake, fake_label)
            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)
            dloss_real = self.criterion_label(dout_real, real_label)

            dloss = (dloss_fake + dloss_real) / 2  # scale the loss to one

            # record loss
            self.record_loss(dloss.detach().cpu().numpy(), optimizer_idx)

            # accuracy hack
            dloss = self.apply_accuracy_hack(dloss, dout_real, dout_fake)
            return dloss

    def generate_tree(self, num_trees=1):
        generator = self.vae.vae_decoder
        return super().generate_tree(generator, num_trees=num_trees)


class GAN(BaseModel):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # number of layer per block
        parser.add_argument("--g_layer", type=int, default=1)
        parser.add_argument("--d_layer", type=int, default=1)
        # optimizer in {"Adam", "SGD"}
        parser.add_argument("--gen_opt", type=str, default="Adam")
        parser.add_argument("--dis_opt", type=str, default="Adam")
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--label_loss", type=str, default="BCELoss")
        # accuracy_hack
        parser.add_argument("--accuracy_hack", type=float, default=1.1)
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

        return BaseModel.add_model_specific_args(parser)

    def __init__(self, config):
        super(GAN, self).__init__(config)
        self.config = config
        self.save_hyperparameters("config")
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type
        self.generator = self.setup_Generator(
            "generator",
            layer_per_block=config.g_layer,
            num_layer_unit=config.gen_num_layer_unit,
            optimizer_option=config.gen_opt,
            learning_rate=config.g_lr,
        )
        self.discriminator = self.setup_Discriminator(
            "discriminator",
            layer_per_block=config.d_layer,
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
        )

        # loss function
        self.criterion_label = self.get_loss_function_with_logit(config.label_loss)

    def forward(self, x):
        # classifier and discriminator
        x = self.generator(x)
        x = F.tanh(x)
        x = self.discriminator(x)
        return x

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0].float()
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        if optimizer_idx == 0:
            ############
            #   generator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))

            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # tree_fake is already computed above
            dout_fake = self.discriminator(tree_fake)
            # generator should generate trees that discriminator think they are real
            gloss = self.criterion_label(dout_fake, real_label)

            # record loss
            self.record_loss(gloss.detach().cpu().numpy(), optimizer_idx)

            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))
            # add noise to generated data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)
            dloss_real = self.criterion_label(dout_real, real_label)

            # fake data (data from generator)
            dout_fake = self.discriminator(
                tree_fake.clone().detach()
            )  # detach so no update to generator
            dloss_fake = self.criterion_label(dout_fake, fake_label)

            # loss function (discriminator classify real data vs generated data)
            dloss = (dloss_real + dloss_fake) / 2

            # record loss
            self.record_loss(dloss.detach().cpu().numpy(), optimizer_idx)

            # accuracy hack
            dloss = self.apply_accuracy_hack(dloss, dout_real, dout_fake)
            return dloss

    def generate_tree(self, num_trees=1):
        generator = self.generator
        return super().generate_tree(generator, num_trees=num_trees)


class GAN_Wloss(GAN):
    @staticmethod
    def add_model_specific_args(parent_parser, resolution=32):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # important argument
        parser.add_argument("--clip_value", type=float, default=0.01)

        return GAN.add_model_specific_args(parser, resolution)

    def configure_optimizers(self):
        config = self.config
        discriminator = self.discriminator

        # clip critic (discriminator) gradient
        # no clip when gp is applied

        clip_value = config.clip_value
        for p in discriminator.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return super().configure_optimizers()

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        # dataset_batch was a list: [array], so just take the array inside
        dataset_batch = dataset_batch[0].float()
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        batch_size = dataset_batch.shape[0]
        # label not used in Wloss
        if optimizer_idx == 0:
            ############
            #   generator
            ############
            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # tree_fake is already computed above
            dout_fake = self.discriminator(tree_fake)

            # generator should maximize dout_fake
            gloss = -dout_fake.mean()

            # record loss
            self.record_loss(gloss.detach().cpu().numpy(), optimizer_idx)

            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)

            # fake data (data from generator)
            dout_fake = self.discriminator(
                tree_fake.clone().detach()
            )  # detach so no update to generator

            # d should maximize diff of real vs fake (dout_real - dout_fake)
            dloss = dout_fake.mean() - dout_real.mean()

            # record loss
            self.record_loss(dloss.detach().cpu().numpy(), optimizer_idx)

            return dloss


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
        dataset_batch = dataset_batch[0].float()
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        batch_size = dataset_batch.shape[0]

        # label not used in Wloss
        if optimizer_idx == 0:
            ############
            #   generator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # tree_fake is already computed above
            dout_fake = self.discriminator(tree_fake)

            # generator should maximize dout_fake
            gloss = -dout_fake.mean()

            # record loss
            self.record_loss(gloss.detach().cpu().numpy(), optimizer_idx)

            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator (and classifier if necessary)
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)

            # fake data (data from generator)
            # detach so no update to generator
            dout_fake = self.discriminator(tree_fake.clone().detach())

            gp = self.gradient_penalty(dataset_batch, tree_fake)
            # d should maximize diff of real vs fake (dout_real - dout_fake)
            dloss = dout_fake.mean() - dout_real.mean() + gp

            # record loss
            self.record_loss(dloss.detach().cpu().numpy(), optimizer_idx)

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
        super(GAN, self).__init__(config)
        self.config = config
        self.save_hyperparameters("config")
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type
        self.generator = self.setup_Generator(
            "generator",
            layer_per_block=config.g_layer,
            num_layer_unit=config.gen_num_layer_unit,
            optimizer_option=config.gen_opt,
            learning_rate=config.g_lr,
            num_classes=config.num_classes,
        )
        self.discriminator = self.setup_Discriminator(
            "discriminator",
            layer_per_block=config.d_layer,
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
        )
        self.classifier = self.setup_Discriminator(
            "classifier",
            layer_per_block=config.d_layer,
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
            output_size=config.num_classes,
        )

        # loss function
        self.criterion_label = self.get_loss_function_with_logit(config.label_loss)
        self.criterion_class = self.get_loss_function_with_logit(config.class_loss)

    def forward(self, x, c):
        # combine x and c into z
        z = self.merge_latent_and_class_vector(x, c)

        # classifier and discriminator
        x = self.generator(z)
        x = F.tanh(x)
        d_predict = self.discriminator(x)
        c_predict = self.classifier(x)
        return d_predict, c_predict

    def training_step(self, dataset_batch, batch_idx, optimizer_idx):
        config = self.config

        dataset_batch, dataset_indices = dataset_batch
        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        dataset_batch = dataset_batch.float()
        dataset_indices = dataset_indices.to(torch.int64)

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        if optimizer_idx == 0:
            ############
            #   generator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            # class vector
            c_fake = (
                torch.randint(0, config.num_classes, (batch_size,))
                .type_as(dataset_batch)
                .to(torch.int64)
            )

            # combine z and c_fake
            z = self.merge_latent_and_class_vector(z, c_fake)

            tree_fake = F.tanh(self.generator(z))
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # tree_fake on Dis
            dout_fake = self.discriminator(tree_fake)
            # generator should generate trees that discriminator think they are real
            gloss_d = self.criterion_label(dout_fake, real_label)

            # tree_fake on Cla
            cout_fake = self.classifier(tree_fake)
            gloss_c = self.criterion_class(cout_fake, c_fake)

            gloss = (gloss_d + gloss_c) / 2

            # record loss
            self.record_loss(gloss.detach().cpu().numpy(), optimizer_idx)

            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            # class vector
            c = (
                torch.randint(0, config.num_classes, (batch_size,))
                .type_as(dataset_batch)
                .to(torch.int64)
            )

            # combine z and c
            z = self.merge_latent_and_class_vector(z, c)

            # detach so no update to generator
            tree_fake = F.tanh(self.generator(z)).clone().detach()
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)
            dloss_real = self.criterion_label(dout_real, real_label)

            # fake data (data from generator)
            dout_fake = self.discriminator(tree_fake)
            dloss_fake = self.criterion_label(dout_fake, fake_label)

            # loss function (discriminator classify real data vs generated data)
            dloss = (dloss_real + dloss_fake) / 2

            # record loss
            self.record_loss(dloss.detach().cpu().numpy(), optimizer_idx)

            # accuracy hack
            dloss = self.apply_accuracy_hack(dloss, dout_real, dout_fake)
            return dloss

        if optimizer_idx == 2:

            ############
            #   classifier
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            # class vector
            c_fake = (
                torch.randint(0, config.num_classes, (batch_size,))
                .type_as(dataset_batch)
                .to(torch.int64)
            )

            # combine z and c
            z = self.merge_latent_and_class_vector(z, c_fake)

            # detach so no update to generator
            tree_fake = F.tanh(self.generator(z)).clone().detach()
            # add noise to data
            tree_fake = self.add_noise_to_samples(tree_fake)

            # fake data (data from generator)
            cout_fake = self.classifier(tree_fake)
            closs_fake = self.criterion_class(cout_fake, c_fake)

            # real data (data from dataloader)
            cout_real = self.classifier(dataset_batch)
            closs_real = self.criterion_class(cout_real, dataset_indices)

            # loss function (discriminator classify real data vs generated data)
            closs = (closs_real + closs_fake) / 2

            # record loss
            self.record_loss(closs.detach().cpu().numpy(), optimizer_idx)

            return closs

    def generate_tree(self, c, num_trees=1):
        config = self.config
        generator = self.generator

        return super(GAN, self).generate_tree(
            generator, c=c, num_classes=config.num_classes, num_trees=num_trees
        )


#####
#   building blocks for models
#####


class Generator(nn.Module):
    def __init__(
        self,
        layer_per_block=1,
        z_size=128,
        output_size=64,
        num_layer_unit=32,
        dropout_prob=0.3,
        spectral_norm=False,
        use_simple_3dgan_struct=False,
        activations=nn.ReLU(True),
    ):
        super(Generator, self).__init__()
        self.z_size = z_size
        self.use_simple_3dgan_struct = use_simple_3dgan_struct

        # layer_per_block must be >= 1
        if layer_per_block < 1:
            layer_per_block = 1

        if use_simple_3dgan_struct:
            self.fc_channel = z_size  # 16
            self.fc_size = 1
        else:
            self.fc_channel = 8  # 16
            self.fc_size = 4

        self.output_size = output_size
        # need int(output_size / self.fc_size) upsampling to increase size
        self.num_blocks = int(np.log2(output_size) - np.log2(self.fc_size))
        if use_simple_3dgan_struct:
            # minus 1 as the final conv block also double the volume
            self.num_blocks = self.num_blocks - 1

        if type(num_layer_unit) is list:
            if len(num_layer_unit) < self.num_blocks:
                raise Exception(
                    "For output_size="
                    + str(output_size)
                    + ", the list of num_layer_unit should have "
                    + str(self.num_blocks)
                    + " elements."
                )
            if len(num_layer_unit) > self.num_blocks:
                num_layer_unit = num_layer_unit[: self.num_blocks]
                print(
                    "For output_size="
                    + str(output_size)
                    + ", the list of num_layer_unit should have "
                    + str(self.num_blocks)
                    + " elements. Trimming num_layer_unit to "
                    + str(num_layer_unit)
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
                if use_simple_3dgan_struct:
                    conv_layer = nn.ConvTranspose3d(
                        num_layer_unit1,
                        num_layer_unit2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                else:
                    conv_layer = nn.ConvTranspose3d(
                        num_layer_unit1, num_layer_unit2, 3, 1, padding=1
                    )

                if spectral_norm:
                    gen_module.append(SpectralNorm(conv_layer))
                else:
                    gen_module.append(conv_layer)
                gen_module.append(nn.BatchNorm3d(num_layer_unit2))
                gen_module.append(activations)
                gen_module.append(nn.Dropout3d(dropout_prob))
                num_layer_unit1 = num_layer_unit2

            if not use_simple_3dgan_struct:
                gen_module.append(nn.Upsample(scale_factor=2, mode="trilinear"))

        # remove tanh for loss with logit
        if use_simple_3dgan_struct:
            conv_layer = nn.ConvTranspose3d(
                num_layer_unit1, 1, kernel_size=4, stride=2, padding=1
            )
        else:
            conv_layer = nn.ConvTranspose3d(num_layer_unit1, 1, 3, 1, padding=1)

        if spectral_norm:
            gen_module.append(SpectralNorm(conv_layer))
        else:
            gen_module.append(conv_layer)
        # gen_module.append(nn.tanh())

        if not self.use_simple_3dgan_struct:
            self.gen_fc = nn.Linear(
                self.z_size,
                self.fc_channel * self.fc_size * self.fc_size * self.fc_size,
            )
        else:
            # create simple layer (not used in training)
            self.gen_fc = nn.Linear(1, 1)

        self.gen = nn.Sequential(*gen_module)

    def forward(self, x):
        if not self.use_simple_3dgan_struct:
            x = self.gen_fc(x)
        x = x.view(
            x.shape[0], self.fc_channel, self.fc_size, self.fc_size, self.fc_size
        )
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        layer_per_block=1,
        output_size=1,
        input_size=64,
        num_layer_unit=16,
        dropout_prob=0.3,
        spectral_norm=False,
        use_simple_3dgan_struct=False,
        activations=nn.LeakyReLU(0.0, True),
    ):
        super(Discriminator, self).__init__()

        self.use_simple_3dgan_struct = use_simple_3dgan_struct

        # layer_per_block must be >= 1
        if layer_per_block < 1:
            layer_per_block = 1

        if use_simple_3dgan_struct:
            self.fc_size = 1
        else:
            self.fc_size = 4  # final height of the volume in conv layers before flatten

        self.input_size = input_size
        self.output_size = output_size
        # need int(input_size / self.fc_size) upsampling to increase size
        # minus 1 for the last conv layer (not in loop so batchnorm/dropout not applied after)
        self.num_blocks = int(np.log2(input_size) - np.log2(self.fc_size))
        if use_simple_3dgan_struct:
            # minus 1 as the final conv block also cut the volume by half
            self.num_blocks = self.num_blocks - 1

        if type(num_layer_unit) is list:
            if len(num_layer_unit) < self.num_blocks:
                raise Exception(
                    "For input_size="
                    + str(input_size)
                    + ", the list of num_layer_unit should have "
                    + str(self.num_blocks)
                    + " elements."
                )
            if len(num_layer_unit) > self.num_blocks:
                num_layer_unit = num_layer_unit[: self.num_blocks]
                print(
                    "For input_size="
                    + str(input_size)
                    + ", the list of num_layer_unit should have "
                    + str(self.num_blocks)
                    + " elements. Trimming num_layer_unit to "
                    + str(num_layer_unit)
                )
            num_layer_unit_list = num_layer_unit
        elif type(num_layer_unit) is int:
            num_layer_unit_list = [num_layer_unit] * self.num_blocks
        else:
            raise Exception("num_layer_unit should be int of list of int.")

        # add initial num_unit to num_layer_unit_list
        num_layer_unit_list = [1] + num_layer_unit_list
        dis_module = []
        #
        for i in range(self.num_blocks):
            num_layer_unit1, num_layer_unit2 = (
                num_layer_unit_list[i],
                num_layer_unit_list[i + 1],
            )

            for _ in range(layer_per_block):
                if use_simple_3dgan_struct:
                    conv_layer = nn.Conv3d(
                        num_layer_unit1,
                        num_layer_unit2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                else:
                    conv_layer = nn.Conv3d(
                        num_layer_unit1, num_layer_unit2, 3, 1, padding=1
                    )

                if spectral_norm:
                    dis_module.append(SpectralNorm(conv_layer))
                else:
                    dis_module.append(conv_layer)

                dis_module.append(nn.BatchNorm3d(num_layer_unit2))
                dis_module.append(activations)
                dis_module.append(nn.Dropout3d(dropout_prob))
                num_layer_unit1 = num_layer_unit2

            if not use_simple_3dgan_struct:
                dis_module.append(nn.MaxPool3d((2, 2, 2)))

        # # remove extra pool layer
        # dis_module = dis_module[:-1]

        if use_simple_3dgan_struct:
            conv_layer = nn.Conv3d(
                num_layer_unit1, self.output_size, kernel_size=4, stride=2, padding=1
            )
        else:
            conv_layer = nn.Conv3d(num_layer_unit1, num_layer_unit1, 3, 1, padding=1)
        dis_module.append(conv_layer)

        self.dis = nn.Sequential(*dis_module)

        if not self.use_simple_3dgan_struct:
            self.dis_fc = nn.Linear(
                num_layer_unit1 * self.fc_size * self.fc_size * self.fc_size,
                self.output_size,
            )
        else:
            # create simple layer (not used in training)
            self.dis_fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.shape[0], -1)
        if not self.use_simple_3dgan_struct:
            x = self.dis_fc(x)
        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        assert encoder.input_size == decoder.output_size
        self.sample_size = decoder.output_size
        self.encoder_z_size = encoder.output_size
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
        f = self.vae_encoder(x)
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
        self.encoder_z_size = encoder.output_size
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
        f = self.vae_encoder(x)
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
