from bugan.functionsPL import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from argparse import Namespace, ArgumentParser
from torch.utils.data import DataLoader, TensorDataset

import pkgutil
import io
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

DEFAULT_NUM_LAYER_UNIT = [256, 512, 256, 128, 64]
DEFAULT_NUM_LAYER_UNIT_REV = DEFAULT_NUM_LAYER_UNIT.copy()
DEFAULT_NUM_LAYER_UNIT_REV.reverse()


class BaseModel(pl.LightningModule):
    """
    Base model
    The parent of all training models
    This model manages settings that shares among all training models, including
    1) setup model components (optimizer, setup on_epoch / on_batch, logging, ...)
    2) wandb logging (log when epoch end)
    3) GAN hacks (label_noise, dropout_prob, ...)
    4) other common functions (generate_trees, get_loss_function_with_logit, create_real_fake_label, ...)
    5) LightningModule functions (configure_optimizers, on_train_epoch_start, on_train_epoch_end)
    *) Note that __init__() and training_step should be implemented in child model
    *) Assumption: the input mesh array is in [-1,1], with -1 means no voxel and 1 means has voxel,
        but most of the time the input is [0,1], and the array is scaled in self.training_step()
    Attributes
    ----------
    model_list : list of nn.Module
        a list containing all model components
            * the items for the same model component should have the same list index
                model_list[i] <-> model_name_list[i] <-> opt_config_list[i] <-> model_ep_loss_list[i]
            * the list index also indicates the optimizer_idx in the training_step
        the elements will be added when setup_Generator()/setup_Discriminator()/setup_VAE() are called
    model_name_list : list of string
        model name string is recorded only for logging purpose
            only used when calling wandbLog() to log loss of each module in training_step()
        the elements will be added when setup_Generator()/setup_Discriminator()/setup_VAE() are called
    opt_config_list : list of tuple
        config for setup optimizer for the components in self.model_list
            the tuple stored is (model_opt_string, model_lr), will be handled by configure_optimizers()
        the elements will be added when setup_Generator()/setup_Discriminator()/setup_VAE() are called
    model_ep_loss_list : list of list of numpy array shape [1]
        record loss of each model component in training_step
            model_ep_loss_list[i][b] is the loss of the model component with index i in training batch with index b
        Please use self.record_loss(loss, optimizer_idx) function to record loss instead of directly append to this list
        the elements will be added when setup_Generator()/setup_Discriminator()/setup_VAE() are called
    other_loss_dict : list of list of numpy array shape [1]
        record custom loss from the model recorded by self.record_loss(loss, loss_name)
        will be processed like model_ep_loss_list (storing list of batch loss, and wandblog mean of them)
    log_reconstruct : boolean
        if True, log input mesh and reconstructed mesh. If False, log sample mesh from random latent vector
    config : Namespace
        dictionary of training parameters
    config.batch_size : int
        the batch_size for the dataloader
    config.resolution : int
        the size of the array for voxelization
        if resolution=32, the resulting array of processing N mesh files are (N, 1, 32, 32, 32)
    config.log_interval : int
        when to log generated sample statistics to wandb
        If the value is 10, the statistics of generated samples will log to wandb per 10 epoch
            (see self.wandbLog(), calculate_log_media_stat())
        The loss values and epoch number are excluded. They will be logged every epoch
    config.log_num_samples : int
        when statistics of generated samples is logged, the number of samples to generate and calculate statistics from
            (see self.wandbLog(), calculate_log_media_stat())
        If config.num_classes is set, the number of generated samples will be (config.num_classes * config.log_num_samples)
    config.label_noise : float
        create label noise by adding uniform random noise to the real/generated labels
        if the value=0.2, the real label will be in [0.8, 1], and fake label will be in [0, 0.2]
    config.z_size : int
        the latent vector size of the model
        latent vector size determines the size of the generated input, and the size of the compressed latent vector in VAE
    config.activation_leakyReLU_slope : float
        the slope of the leakyReLU activation (see torch leakyReLU)
        leakyReLU activation is used for all layers except the last layer of all models
    config.dropout_prob : float
        the dropout probability of all models (see torch Dropout3D)
        all generator/discriminator use (ConvT/Conv)-BatchNorm-activation-dropout structure
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """

        # for config argument, see model description above
        default_model_config = {
            "batch_size": 4,  # number of data per batch, B.
            "resolution": 64,  # size of data, res. data shape: (B,1,res,res,res)
            "log_interval": 10,  # the interval (in epoch) to log generated samples and save ckpt
            "log_num_samples": 3,  # in log_interval epoch, the number of log generated samples
            "label_noise": 0.0,  # label smoothing to the discriminator/classifier lables (from {0,1} to {label_noise, 1-label_noise})
            "z_size": 128,  # the latent vector size for GAN, VAE, and other models with similar structures
            "activation_leakyReLU_slope": 0.01,  # the slope of leakyReLU in Generator/Discriminator (see Generator/Discriminator)
            "dropout_prob": 0.0,  # the dropout probability in the Generator/Discriminator (see Generator/Discriminator)
            "kernel_size": 5,  # kernel size of convT/conv layer in the Generator/Discriminator (see Generator/Discriminator)
            "fc_size": 2,  # the data shape (B,layer_unit[k],fc,fc,fc) between convT/conv layer and fc_layer (see Generator/Discriminator)
        }

        default_model_config.update(vars(config))
        # remove config.config as it is just a copy
        if hasattr(config, "config"):
            config.pop("config")
        config_rev = Namespace(**default_model_config)
        return config_rev

    @staticmethod
    def combine_namespace(base, update):
        """
        helper function to combine Namespace object
        values in base will be overwritten by values in update
        Parameters
        ----------
        base : Namespace
        update : Namespace
        Returns
        -------
        Namespace
            the Namespace with default arguments from base replaced by arguments in update
        """
        base = vars(base)
        base.update(vars(update))
        return Namespace(**base)

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
        # record other losses specified by the record_loss function in training_step
        self.other_loss_dict = {}

        # whether to log input mesh and reconstructed mesh instead of sample mesh from random z
        self.log_reconstruct = False

    def setup_Generator(
        self,
        model_name,
        num_layer_unit,
        optimizer_option,
        learning_rate,
        num_classes=None,
        kernel_size=3,
        fc_size=2,
    ):
        """
        function to set Generator for the Model
        this function will be used in ChildModel to set up Generator, its optimizer, and wandbLog data
        Parameters
        ----------
        model_name : string
            model name string is recorded only for logging purpose
            only used when calling wandbLog() to log loss of each module in training_step()
        num_layer_unit : int/list
            the number of unit in the ConvT layer
            if int, every ConvT will have the same specified number of unit.
            if list, every ConvT layer between the upsampling layers will have the specified number of unit.
            see also Generator class
        optimizer_option : string
            the string in ['Adam', 'SGD'], setup the optimizer of the Generator
        learning_rate : float
            the learning_rate of the Generator optimizer
        num_classes : int/None
            if None, unconditional data. The input vector size is the same as latent vector size
            if int, conditional data. The input vector size = latent vector size + number of classes
                (class vector assume to be one-hot)
        kernel_size : int
            the kernel size of the convT layer. padding will be adjusted so the output_size of the model is not affected
        fc_size : int
            the size of the input volume for the first convT layer.
            For fc_size=2, the last fc layer will output B*unit_list[0]*fc_size**3, and reshape into (B,unit_list[0],fc_size, fc_size, fc_size).
            The lower the value, the number of upsampling layer and convT layer increases
            (number of convTLayer = int(np.log2(self.resolution)) - int(np.log2(self.fc_size))).
        Returns
        -------
        generator : nn.Module
        """
        config = self.config
        if not num_classes:
            z_size = config.z_size
        else:
            z_size = config.z_size + num_classes

        generator = Generator(
            z_size=z_size,
            resolution=config.resolution,
            num_layer_unit=num_layer_unit,
            kernel_size=kernel_size,
            fc_size=fc_size,
            dropout_prob=config.dropout_prob,
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
        num_layer_unit,
        optimizer_option,
        learning_rate,
        output_size=1,
        kernel_size=3,
        fc_size=2,
    ):
        """
        function to set Discriminator for the Model
        this function will be used in ChildModel to set up Discriminator, its optimizer, and wandbLog data
        Parameters
        ----------
        model_name : string
            model name string is recorded only for logging purpose
            only used when calling wandbLog() to log loss of each module in training_step()
        num_layer_unit : int/list
            the number of unit in the Conv layer
            if int, every Conv will have the same specified number of unit.
            if list, every Conv layer between the pooling layers will have the specified number of unit.
            see also Discriminator class
        optimizer_option : string
            the string in ['Adam', 'SGD'], setup the optimizer of the Discriminator
        learning_rate : float
            the learning_rate of the Discriminator optimizer
        output_size : int
            the final output vector size. see also Discriminator class
        kernel_size : int
            the kernel size of the conv layer. padding will be adjusted so the output_size of the model is not affected
        fc_size : int
            the size of the output volume for the last conv layer.
            For fc_size=2, the last conv layer will output (B,unit_list[-1],fc_size, fc_size, fc_size),
            and flatten that for fc_layer.The lower the value, the number of downsampling layer and conv layer increases
            (number of ConvLayer = int(np.log2(self.resolution)) - int(np.log2(self.fc_size))).
        Returns
        -------
        discriminator : nn.Module
        """
        config = self.config

        discriminator = Discriminator(
            output_size=output_size,
            resolution=config.resolution,
            num_layer_unit=num_layer_unit,
            kernel_size=kernel_size,
            fc_size=fc_size,
            dropout_prob=config.dropout_prob,
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
        encoder_num_layer_unit,
        decoder_num_layer_unit,
        optimizer_option,
        learning_rate,
        num_classes=None,
        kernel_size=[3, 3],
        fc_size=[2, 2],
    ):
        """
        function to set VAE for the Model
        this function will be used in ChildModel to set up VAE, its optimizer, and wandbLog data
        Note that the VAE is made by a generator and a discriminator, so this function shares
            lots of similarities with the setup_Discriminator() and setup_Generator()
        Parameters
        ----------
        model_name : string
            model name string is recorded only for logging purpose
            only used when calling wandbLog() to log loss of each module in training_step()
        encoder_num_layer_unit : int/list
            same as the num_layer_unit for setup_Discriminator()
            see also Discriminator class
        decoder_num_layer_unit : int/list
            same as the num_layer_unit for setup_Generator()
            see also Generator class
        optimizer_option : string
            the string in ['Adam', 'SGD'], setup the optimizer of the VAE
        learning_rate : float
            the learning_rate of the VAE optimizer
        num_classes : int/None
            if None, unconditional data. The input vector size is the same as latent vector size
            if int, conditional data. The input vector size = latent vector size + number of classes
                (class vector assume to be one-hot)
        kernel_size : int
            the kernel size of the conv/convT layer for discriminator/generator. See setup_Discriminator/Generator above.
        fc_size : int
            the size of the output volume for the last conv layer/ input volume for the first convT layer. See setup_Discriminator/Generator above.
        Returns
        -------
        vae : nn.Module
        """
        config = self.config

        encoder = Discriminator(
            output_size=config.z_size,
            resolution=config.resolution,
            num_layer_unit=encoder_num_layer_unit,
            kernel_size=kernel_size[0],
            fc_size=fc_size[0],
            dropout_prob=config.dropout_prob,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        decoder_input_size = config.z_size
        if num_classes is not None:
            decoder_input_size = decoder_input_size + num_classes

        decoder = Generator(
            z_size=decoder_input_size,
            resolution=config.resolution,
            num_layer_unit=decoder_num_layer_unit,
            kernel_size=kernel_size[1],
            fc_size=fc_size[1],
            dropout_prob=config.dropout_prob,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        vae = VAE(
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            dim_class_embedding=num_classes,
        )
        # setup component in __init__() lists
        # for configure_optimizers() and record loss
        self.setup_model_component(vae, model_name, optimizer_option, learning_rate)
        return vae

    # TODO: doc for ZVAEGAN
    def setup_VAE_mod(
        self,
        model_name,
        encoder_num_layer_unit,
        decoder_num_layer_unit,
        optimizer_option,
        learning_rate,
        num_classes=None,
        kernel_size=[3, 3],
        fc_size=[2, 2],
        class_std=1,
    ):
        config = self.config

        encoder = Discriminator(
            output_size=config.z_size,
            resolution=config.resolution,
            num_layer_unit=encoder_num_layer_unit,
            kernel_size=kernel_size[0],
            fc_size=fc_size[0],
            dropout_prob=config.dropout_prob,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        decoder = Generator(
            z_size=config.z_size,
            resolution=config.resolution,
            num_layer_unit=decoder_num_layer_unit,
            kernel_size=kernel_size[1],
            fc_size=fc_size[1],
            dropout_prob=config.dropout_prob,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        vae = VAE_mod(
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            class_std=class_std,
        )
        # setup component in __init__() lists
        # for configure_optimizers() and record loss
        self.setup_model_component(vae, model_name, optimizer_option, learning_rate)
        return vae

    def setup_model_component(self, model, model_name, model_opt_string, model_lr):
        """
        setup model components to the lists for later use
        record (model_list, opt_config_list) for configure_optimizers()
        record (model_name_list, model_ep_loss_list) for logging loss
        * this function will be called when initializing model components
            in setup_Generator/Discriminator/VAE
        Parameters
        ----------
        model : nn.Module
            the model component to setup
        model_name : string
            model name string is recorded only for logging purpose
            only used when calling wandbLog() to log loss of each module in training_step()
        model_opt_string : string
            the string in ['Adam', 'SGD'], setup the optimizer of the model
        model_lr : float
            the learning_rate of the model optimizer
        """
        self.model_list.append(model)
        self.model_name_list.append(model_name)
        self.opt_config_list.append((model_opt_string, model_lr))
        self.model_ep_loss_list.append([])

    def get_loss_function_with_logit(self, loss_option):
        """
        return torch loss function based on the string loss_option
        the returned loss assume input to be logit (before sigmoid/tanh)
        the returned loss reduction method: mean over batch, sum over other dimenisons
        Parameters
        ----------
        loss_option : string
            loss_option in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        Returns
        -------
        loss : nn Loss Function
            the corresponding loss function based on the loss_option string
        """

        if loss_option == "BCELoss":
            loss = (
                lambda gen, data: nn.BCEWithLogitsLoss(reduction="sum")(gen, data)
                / data.shape[0]
            )
        elif loss_option == "MSELoss":
            loss = (
                lambda gen, data: torch.sum(
                    nn.MSELoss(reduction="none")(F.tanh(gen), data)
                )
                / data.shape[0]
            )
        elif loss_option == "CrossEntropyLoss":
            # remake CE loss with BCELoss for applying label_noise
            def mod_CELoss(gen, data):
                c = data
                c = c.reshape((-1, 1))
                # check for class_index >= 0
                c_mask = (c >= 0).type_as(c)
                c = c * c_mask
                # to onehot
                c_onehot = torch.zeros(gen.shape).type_as(c)
                c_onehot = c_onehot.scatter(1, c, 1)
                data = c_onehot.type_as(gen)
                # apply label loss
                noise = torch.rand(gen.shape).type_as(gen) * self.config.label_noise
                data = data * (1 - noise) + (1 - data) * noise
                # apply loss
                loss_fn = nn.BCEWithLogitsLoss(reduction="none")
                loss_value = loss_fn(gen, data)
                # ignore those class_index < 0
                loss_value = torch.sum(torch.mean(loss_value * c_mask, 1)) / torch.sum(
                    c_mask
                )
                return loss_value

            loss = mod_CELoss

            # loss = lambda gen, data: nn.CrossEntropyLoss(
            #     ignore_index=-1, reduction="sum"
            # )(gen, data) / torch.sum(data >= 0)
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
        """
        default function for pl.LightningModule to setup optimizer
        will be called automatically in pl.Trainer.fit()
        The optimizers will be setup based on the self.opt_config_list on
            the model components in self.model_list.
            (self.opt_config_list[i] for the optimizer of the component self.model_list[i])
        if the attribute self.config.cyclicLR_magnitude exists, also set cyclicLR schedulers
        Parameters
        ----------
        self.model_list : list of nn.Module
        self.opt_config_list : list of tuple
        self.config.cyclicLR_magnitude : float
            if exists, set all model optimizers with cyclicLR scheduler
            the base_lr = model_lr / cyclicLR_magnitude
            the max_lr = model_lr * cyclicLR_magnitude
        Returns
        -------
        optimizer_list : list of torch.optim optimizer
            the optimizers of all model componenets
        scheduler_list : list of torch.optim.lr_scheduler
            the cyclicLR schedulers of all model componenets if self.config.cyclicLR_magnitude attribute exists
        """
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

    #
    def get_model_optimizer(self, model, optimizer_option, lr):
        """
        return torch optimizer based on the string optimizer_option and the learning rate
        Parameters
        ----------
        model : nn.Module
            the model component to setup optimizer
        optimizer_option : string
            the string in ['Adam', 'SGD'], setup the optimizer of the model
        lr : float
            the learning_rate of the model optimizer
        Returns
        -------
        torch.optim Optimizer
            the optimizer of the model component
        """

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
        """
        default function for pl.LightningModule to run on the start of every train epoch
        here the function just setup for recording the loss of model components

        Parameters
        ----------
        self.model_list : list of nn.Module
        self.model_ep_loss_list : list of list of numpy array shape [1]
            model_ep_loss_list[i][b] is the loss of the model component with index i in training batch with index b
        """

        # reset ep_loss
        # set model to train
        for idx in range(len(self.model_ep_loss_list)):
            self.model_ep_loss_list[idx] = []
            self.model_list[idx].train()
        # also process other_loss_dict like model_ep_loss_list
        for i in self.other_loss_dict:
            self.other_loss_dict[i] = []

    #####
    #   on_train_epoch_end()
    #####
    def on_train_epoch_end(self, epoch_output=None):
        """
        default function for pl.LightningModule to run on the end of every train epoch
        here the function just setup the log information as dictionary and log it to wandb
        Parameters
        ----------
        self.current_epoch : int
            default pl.LightningModule attribute storing the number of current_epoch
        self.model_name_list : list of string
            model name string to construct dictionary key to log loss to wandb
        self.model_ep_loss_list : list of list of numpy array shape [1]
            the loss data to log to wandb
        self.config.log_interval : int
            when to log generated sample statistics to wandb
            If the value is 10, the statistics of generated samples will log to wandb per 10 epoch
                (see self.wandbLog(), calculate_log_media_stat())
            The loss values and epoch number are excluded. They will be logged every epoch
        self.config.log_num_samples : int
            when statistics of generated samples is logged, the number of samples to generate and calculate statistics from
                (see self.wandbLog(), calculate_log_media_stat())
            If config.num_classes is set, the number of generated samples will be (config.num_classes * config.log_num_samples)
        self.trainer.datamodule.class_list : list of string
            using pl.LightningModule link to trainer and then reach datamodule to obtain the name of classes
        """

        log_dict = {"epoch": self.current_epoch}

        # record loss and add to log_dict
        for idx in range(len(self.model_ep_loss_list)):
            loss = np.mean(self.model_ep_loss_list[idx])
            loss_name = "Loss/" + self.model_name_list[idx]
            log_dict[loss_name] = loss
        # record loss for other_loss_dict like model_ep_loss_list
        for i in self.other_loss_dict:
            loss = np.mean(self.other_loss_dict[i])
            log_dict["Loss/" + str(i)] = loss

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

    def wandbLog(
        self, class_list=None, initial_log_dict={}, log_media=False, log_num_samples=1
    ):
        """
        add data/statistics of generated meshes into the log_dict and log information to wandb
        image/mesh/statistics are calculated by calculate_log_media_stat() in functionPL.py
        mesh data statistics
        1) average number of voxel per tree
        2) average number of voxel cluster per tree (check distance function)
        3) images of all generated tree
        4) meshes of all generated tree
        5) mean of per voxel std over generated trees
        Parameters
        ----------
        class_list : list of string
            the name of classes (to log to wandb)
        initial_log_dict : dict
            dictionary containing information to log to wandb
            this function add more data/statistics to it and log it to wandb
        log_media : boolean
            whether to generate meshes and log images/3Dobjects/statistics of the generated meshes to wandb
        log_num_samples : int
            if log_media=True, how many meshes to generate
            if class_list is not None (conditional model), the number is the number of meshes generated for each class
                the total number of generated meshes will be (log_num_samples * len(class_list))
        """

        if log_media:

            if class_list is not None:
                num_classes = len(class_list)
                # log condition model data
                for c in range(num_classes):
                    sample_trees = self.generate_tree(c=c, num_trees=16)
                    (
                        numpoints,
                        num_cluster,
                        image,
                        voxelmesh,
                        std,
                    ) = calculate_log_media_stat(sample_trees, log_num_samples)

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
                if self.log_reconstruct:
                    # for VAE, log both input and reconstructed samples instead of just generated samples
                    # sample only 1 batch
                    sample_input = self.trainer.datamodule.sample_data(num_samples=16)
                    (
                        numpoints,
                        num_cluster,
                        image,
                        voxelmesh,
                        std,
                    ) = calculate_log_media_stat(sample_input, log_num_samples)
                    # log input
                    initial_log_dict["input_numpoints"] = numpoints
                    initial_log_dict["input_num_cluster"] = num_cluster
                    initial_log_dict["input_image"] = image
                    initial_log_dict["input_voxelmesh"] = voxelmesh
                    initial_log_dict["input_per_voxel_std"] = std

                    # generate reconstructed samples
                    sample_input = torch.unsqueeze(torch.Tensor(sample_input), 1)
                    sample_input = sample_input.type_as(
                        self.vae.vae_decoder.gen_fc.weight
                    )
                    sample_trees = self.forward(sample_input)
                    sample_trees = sample_trees[:, 0, :, :, :].detach().cpu().numpy()
                else:
                    sample_trees = self.generate_tree(num_trees=16)

                (
                    numpoints,
                    num_cluster,
                    image,
                    voxelmesh,
                    std,
                ) = calculate_log_media_stat(sample_trees, log_num_samples)

                # add list record to log_dict
                initial_log_dict["sample_numpoints"] = numpoints
                initial_log_dict["sample_num_cluster"] = num_cluster
                initial_log_dict["sample_image"] = image
                initial_log_dict["sample_voxelmesh"] = voxelmesh
                initial_log_dict["sample_per_voxel_std"] = std

        wandb.log(initial_log_dict)

    #####
    #   training_step() related function
    #####

    def training_step(self, dataset_batch, batch_idx, optimizer_idx=0):
        """
        default function for pl.LightningModule to train the model
        the model takes the dataset_batch to calculate the loss of the model components
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input data batch from the datamodule in datamodule_process class
            if the data is unconditional, dataset_batch is in the form [array]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            if the data is conditional, dataset_batch is in the form [array, index]
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
            see datamodulePL.py datamodule_process class
        batch_idx : int
            the index of the batch in datamodule
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
        Returns
        -------
        loss : torch.Tensor of shape [1]
            the loss of the model component.
        """
        config = self.config

        dataset_indices = None
        if hasattr(self.config, "num_classes") and self.config.num_classes > 0:
            # dataset_batch was a list: [array, index]
            dataset_batch, dataset_indices = dataset_batch
            dataset_batch = dataset_batch.float()
            dataset_indices = dataset_indices.to(torch.int64)
        else:
            # dataset_batch was a list: [array], so just take the array inside
            dataset_batch = dataset_batch[0].float()

        # scale to [-1,1]
        dataset_batch = dataset_batch * 2 - 1

        # calculate loss for model components
        loss = self.calculate_loss(dataset_batch, dataset_indices, optimizer_idx)

        # check loss is not NaN (raise Exception if isNaN)
        assert not torch.any(torch.isnan(loss))
        # record loss
        self.record_loss(loss.detach().cpu().numpy(), optimizer_idx)
        return loss

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where array is in shape (B,). B = config.batch_size
                each element is the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
        Returns
        -------
        loss : torch.Tensor of shape [1]
            the loss of the model component.
        """
        pass

    def create_real_fake_label(self, dataset_batch):
        """
        create true/false label for discriminator loss
            normally, true label is 1 and false label is 0
            this function also add smoothing threshold to the labels,
            so true: 1-label_noise, false: 0+label_noise
        Parameters
        ----------
        dataset_batch : torch.Tensor
            data from the dataloader. just for obtaining bach_size and tensor type
        self.config.label_noise : float
            create label smoothing by adding constant to the real/generated labels
            if the value=0.2, the real label will be in 0.8, and fake label will be in 0.2
        Returns
        -------
        real_label : torch.Tensor
            the labels for Discriminator on classifying data from dataset
            the label is scaled to (1-label_noise), occassionally flipped to (label_noise)
        fake_label : torch.Tensor
            the labels for Discriminator on classifying the generated data from the model
            the label is scaled to (label_noise), occassionally flipped to (1-label_noise)
        """

        # TODO: refactor to not use dataset_batch, just batch_size
        config = self.config
        batch_size = dataset_batch.shape[0]
        # labels
        # soft label
        # modified scale to [1-label_noise,1]
        # modified scale to [0,label_noise]
        # real_label = 1 - (torch.rand(batch_size) * config.label_noise)
        # fake_label = torch.rand(batch_size) * config.label_noise
        real_label = torch.ones(batch_size) * (1 - config.label_noise)
        fake_label = torch.ones(batch_size) * config.label_noise

        real_label = torch.unsqueeze(real_label, 1).float().type_as(dataset_batch)
        fake_label = torch.unsqueeze(fake_label, 1).float().type_as(dataset_batch)
        return real_label, fake_label

    def record_loss(self, loss, optimizer_idx=0, loss_name=None):
        """
        save loss to list for updating loss on wandb log
        this function will be called in the training_step()
        Note that the optimizer_idx is also the model component index in self.model_list
        if the loss_name is set, new log item will be added instead of replacing any of the loss
            of the model components. the optimizer_idx will be ignored.
        Parameters
        ----------
        loss : numpy array shape [1]
            the loss calculated for the model component
        optimizer_idx : int
            the index of the optimizer called in training_step()
            this is also the model index in self.model_list
        loss_name : string
            the name of the custom loss that should record to wandb
            the loss will be processed like model_ep_loss_list
            (record list of batch loss and wandblog mean of the loss)
        """
        if loss_name:
            if loss_name in self.other_loss_dict:
                self.other_loss_dict[loss_name].append(loss)
            else:
                self.other_loss_dict[loss_name] = [loss]
        else:
            self.model_ep_loss_list[optimizer_idx].append(loss)

    @staticmethod
    def merge_latent_and_class_vector(
        latent_vector, class_vector, num_classes, embedding_fn=None
    ):
        """
        for conditional models,
        if embedding_fn=None:
            given latent_vector (B, Z) and class_vector (B),
            reshape class_vector to one-hot (B, num_classes),
            and merge with latent_vector
        else:
            process class_vector using embedding_fn(class_vector)
            and merge with latent_vector
        Parameters
        ----------
        latent_vector : torch.Tensor
            the latent vector with shape (B, Z)
        class_vector : torch.Tensor
            the class vector with shape (B,), where each value is the class index integer
        num_classes : int
        embedding_fn : function similar to torch.nn.Embedding or None
            if None: class_vector will turn into one-hot encoding and merge with latent_vector
            else: embedding_fn(class_vector) and merge with latent_vector

        Returns
        -------
        z : torch.Tensor
            if embedding_fn=None, the merged latent vector with shape (B, Z + num_classes)
            else: the merged latent vector with shape (B, Z + num_embeddings)

        """
        z = latent_vector
        c = class_vector
        batch_size = z.shape[0]

        if embedding_fn is not None:
            c_onehot = embedding_fn(c)
        else:
            # convert c to one-hot
            c = c.reshape((-1, 1))
            c_onehot = torch.zeros([batch_size, num_classes]).type_as(c)
            c_onehot = c_onehot.scatter(1, c, 1)

        c_onehot = c_onehot.type_as(z)

        # merge with z to be generator input
        z = torch.cat((z, c_onehot), 1)
        return z

    def generate_tree(
        self, generator, c=None, num_classes=None, embedding_fn=None, num_trees=1
    ):
        """
        generate tree
        for unconditional model, this takes the generator of the model and generate trees
        for conditional model, this also take class index c and num_classes,
            to generate trees of the class with index c
        this function will generate n trees per call (n = num_trees),
            each tree is in shape [res, res, res] (res = config.resolution)
        Parameters
        ----------
        generator : nn.Module
            the generator model component to generate samples
        c : int
            the class index of the class to generate
                the class index is based on the datamodule.class_list
                see datamodulePL.py datamodule_process class
        num_classes : int
            the total number of classes in dataset
        num_trees : int
            the number of trees to generate
        Returns
        -------
        result : numpy.ndarray shape [num_trees, res, res, res]
            the generated samples
        """
        batch_size = self.config.batch_size
        resolution = generator.resolution

        if batch_size > num_trees:
            batch_size = num_trees

        result = None

        num_runs = int(np.ceil(num_trees / batch_size))
        # ignore discriminator
        for i in range(num_runs):

            if c is not None:
                # generate noise vector
                z = torch.randn(batch_size, self.config.z_size).type_as(
                    generator.gen_fc.weight
                )
                # turn class vector the same device as z, but with dtype Long
                c = torch.ones(batch_size) * c
                c = c.type_as(z).to(torch.int64)

                # combine z and c
                z = self.merge_latent_and_class_vector(
                    z,
                    c,
                    num_classes,
                    embedding_fn=embedding_fn,
                )
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
    """
    VAE
    This model contains an encoder and a decoder
    data will be processed by the encoder to be a latent vector, and the decoder will
        take the latent vector to reconstruct the data back.
    The model will train on the reconstruction loss between the original data and the reconstructed data
    Attributes
    ----------
    config : Namespace
        dictionary of training parameters, should also contains parameters needed for BaseModel
    config.vae_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the VAE
        Using 'Adam' may cause VAE KL loss go to inf.
    config.vae_lr : float
        the learning_rate of the VAE
    config.kl_coef : float
        the coefficient of the KL loss in the final VAE loss
    config.decoder_num_layer_unit : int/list
        the decoder_num_layer_unit for BaseModel setup_VAE()
        see also Generator class, BaseModel setup_VAE()
    config.encoder_num_layer_unit : int/list
        the encoder_num_layer_unit for BaseModel setup_VAE()
        see also Discriminator class, BaseModel setup_VAE()
    self.vae : nn.Module
        the model component from setup_VAE()
    self.criterion_reconstruct : nn Loss function
        the loss function based on MSELoss
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """
        config = super(VAE_train, VAE_train).setup_config_arguments(config)

        # for config argument, see model description above
        default_model_config = {
            "vae_opt": "Adam",
            "vae_lr": 1e-5,
            "kl_coef": 1,
            "decoder_num_layer_unit": [128, 256, 512, 256, 128, 128],
            "encoder_num_layer_unit": [128, 128, 128, 256, 256, 128],
        }

        default_model_config.update(vars(config))
        config_rev = Namespace(**default_model_config)
        return config_rev

    def __init__(self, config):
        super(VAE_train, self).__init__(config)
        # assert(vae.sample_size == discriminator.input_size)
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config
        self.save_hyperparameters("config")

        kernel_size = multiple_components_param(config.kernel_size, 2)
        fc_size = multiple_components_param(config.fc_size, 2)

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type
        self.vae = self.setup_VAE(
            model_name="VAE",
            encoder_num_layer_unit=config.encoder_num_layer_unit,
            decoder_num_layer_unit=config.decoder_num_layer_unit,
            optimizer_option=config.vae_opt,
            learning_rate=config.vae_lr,
            kernel_size=kernel_size,
            fc_size=fc_size,
        )

        # log input mesh and reconstructed mesh
        self.log_reconstruct = True

        self.criterion_reconstruct = (
            lambda gen, data: torch.sum(
                nn.MSELoss(reduction="none")(F.tanh(gen), data)
                * self.calculate_pos_weight(data)
            )
            / data.shape[0]
        )

    def calculate_pos_weight(self, data):
        """
        calc pos_weight (negative weigth = 1, positive weight = num_zeros/num_ones)
        assume MSELoss: data in [-1,1]
        Parameters
        ----------
        data : torch.Tensor
            the input data tensor from the dataset_batch
            if the datamodule is in datamodule_process class,
                the input should be unconditional, int the form [array]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
        Returns
        -------
        pos_weight : torch.Tensor
            the shape of pos_weight for MSELoss is (B, 1, res, res, res)
            (negative weigth = 1, positive weight = num_zeros/num_ones)
        """
        target = (data + 1) / 2
        #
        num_ones = torch.sum(target)
        num_zeros = torch.sum(1 - target)
        pos_weight = num_zeros / num_ones - 1
        pos_weight = (target * pos_weight) + 1
        return pos_weight

    def forward(self, x):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input data tensor
            if the datamodule is in datamodule_process class,
                the input should be unconditional, int the form [array]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
        Returns
        -------
        x : torch.Tensor of shape (B, 1, res, res, res)
            the reconstructed data from the VAE based on input x
        """
        x = self.vae(x)
        return x

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
        self.criterion_reconstruct : nn Loss function
            the loss function based on MSELoss to calculate the loss of VAE
        Returns
        -------
        vae_loss : torch.Tensor of shape [1]
            the loss of the VAE.
        """
        config = self.config

        batch_size = dataset_batch.shape[0]

        reconstructed_data, z, mu, logVar = self.vae(dataset_batch, output_all=True)

        vae_rec_loss = self.criterion_reconstruct(reconstructed_data, dataset_batch)
        self.record_loss(vae_rec_loss.detach().cpu().numpy(), loss_name="rec_loss")

        # add KL loss
        KL = self.vae.calculate_log_prob_loss(z, mu, logVar) * config.kl_coef
        self.record_loss(KL.detach().cpu().numpy(), loss_name="KL loss")

        vae_loss = vae_rec_loss + KL

        return vae_loss

    def generate_tree(self, num_trees=1):
        """
        the function to generate tree
        this function specifies the generator module of this model and pass to the parent generate_tree()
            see BaseModel generate_tree()
        Parameters
        ----------
        num_trees : int
            the number of trees to generate
        Returns
        -------
        result : numpy.ndarray shape [num_trees, res, res, res]
            the generated samples
        """
        generator = self.vae.vae_decoder
        return super().generate_tree(generator, num_trees=num_trees)


class VAEGAN(BaseModel):
    """
    VAEGAN
    This model contains an encoder, a decoder, and a discriminator
    data will be processed by the encoder to be a latent vector, and the decoder will
        take the latent vector to reconstruct the data back. Finally, the discriminator
        will give the score of the reconstructed data on how close it is to real data
    The VAE part of the model will train on the reconstruction loss between the original data and the reconstructed data
    The discriminator part will train on prediction loss on classifying reconstructed data and real data
    reference: https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner/blob/master/CVAE-GAN/CVAE-GAN.py
    this model is based on CVAEGAN, but with classifier removed and train on unconditional data
    Attributes
    ----------
    config : Namespace
        dictionary of training parameters
    config.vae_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the VAE
        Using 'Adam' may cause VAE KL loss go to inf.
    config.dis_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the Discriminator
    config.label_loss : string
        label_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assuming input to be logit (before sigmoid/tanh)
        this is the prediction loss for discriminator on classifying reconstructed data and real data
    config.vae_lr : float
        the learning_rate of the VAE
    config.d_lr : float
        the learning_rate of the Discriminator
    config.kl_coef : float
        the coefficient of the KL loss in the final VAE loss
    config.d_rec_coef : float
        the coefficient of the Discriminator loss compared to the reconstruction loss
    config.FMrec_coef : float
        the coefficient of the Discriminator Feature Matching between real and reconstructed
        This FM match the feature of each real object to its reconstructed object.
        This part works together with rec_loss for VAE reconstruction
    config.FMgan_coef : float
        the coefficient of the Discriminator Feature Matching between real and generated
        This FM match the mean feature over real object to the mean feature over reconstructed object.
        This mean feature matching works to imporve GAN performance.
    config.decoder_num_layer_unit : int/list
        the decoder_num_layer_unit for BaseModel setup_VAE()
        see also Generator class, BaseModel setup_VAE()
    config.encoder_num_layer_unit : int/list
        the encoder_num_layer_unit for BaseModel setup_VAE()
        see also Discriminator class, BaseModel setup_VAE()
    config.dis_num_layer_unit : int/list
        the num_layer_unit for BaseModel setup_Discriminator()
        see also Discriminator class, BaseModel setup_Discriminator()
    self.vae : nn.Module
        the model component from setup_VAE()
    self.discriminator : nn.Module
        the model component from setup_Discriminator()
    self.criterion_label : nn Loss function
        the loss function based on config.label_loss
    self.criterion_reconstruct : nn Loss function
        the loss function based on MSELoss
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """
        config = super(VAEGAN, VAEGAN).setup_config_arguments(config)

        # for config argument, see model description above
        default_model_config = {
            "vae_opt": "Adam",
            "dis_opt": "Adam",
            "label_loss": "BCELoss",
            "vae_lr": 1e-5,
            "d_lr": 1e-5,
            "kl_coef": 1,
            "d_rec_coef": 1,
            "FMrec_coef": 0,
            "FMgan_coef": 0,
            "decoder_num_layer_unit": [128, 256, 512, 256, 128, 128],
            "encoder_num_layer_unit": [128, 128, 128, 256, 256, 128],
            "dis_num_layer_unit": [128, 128, 128, 256, 256, 128],
        }

        default_model_config.update(vars(config))
        config_rev = Namespace(**default_model_config)
        return config_rev

    def __init__(self, config):
        super(VAEGAN, self).__init__(config)
        # assert(vae.sample_size == discriminator.input_size)
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config
        self.save_hyperparameters("config")

        kernel_size = multiple_components_param(config.kernel_size, 3)
        fc_size = multiple_components_param(config.fc_size, 3)

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type
        self.vae = self.setup_VAE(
            model_name="VAE",
            encoder_num_layer_unit=config.encoder_num_layer_unit,
            decoder_num_layer_unit=config.decoder_num_layer_unit,
            optimizer_option=config.vae_opt,
            learning_rate=config.vae_lr,
            kernel_size=kernel_size[:2],
            fc_size=fc_size[:2],
        )
        self.discriminator = self.setup_Discriminator(
            "discriminator",
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
            kernel_size=kernel_size[2],
            fc_size=fc_size[2],
        )

        # loss function
        self.criterion_label = self.get_loss_function_with_logit(config.label_loss)
        self.criterion_reconstruct = (
            lambda gen, data: torch.sum(
                nn.MSELoss(reduction="none")(F.tanh(gen), data)
                * self.calculate_pos_weight(data)
            )
            / data.shape[0]
        )

        self.criterion_FM = self.get_loss_function_with_logit("MSELoss")

    def calculate_pos_weight(self, data):
        """
        calc pos_weight (negative weigth = 1, positive weight = num_zeros/num_ones)
        assume MSELoss: data in [-1,1]
        Parameters
        ----------
        data : torch.Tensor
            the input data tensor from the dataset_batch
            if the datamodule is in datamodule_process class,
                the input should be unconditional, int the form [array]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
        Returns
        -------
        pos_weight : torch.Tensor
            the shape of pos_weight for MSELoss is (B, 1, res, res, res)
            (negative weigth = 1, positive weight = num_zeros/num_ones)
        """

        target = (data + 1) / 2
        #
        num_ones = torch.sum(target)
        num_zeros = torch.sum(1 - target)
        pos_weight = num_zeros / num_ones - 1
        pos_weight = (target * pos_weight) + 1
        return pos_weight

    def forward(self, x):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input data tensor
            if the datamodule is in datamodule_process class,
                the input should be unconditional, int the form [array]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
        Returns
        -------
        x : torch.Tensor of shape (B, 1)
            the discriminator score/logit of the reconstructed data from the VAE part
                to show how close the reconstructed data is looking like real data
        """
        # VAE
        x = self.vae(x)
        x = F.tanh(x)
        # discriminator
        x = self.discriminator(x)
        return x

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
            here self.vae=0, self.discriminator=1
        self.criterion_label : nn Loss function
            the loss function based on config.label_loss to calculate the loss of discriminator
        self.criterion_reconstruct : nn Loss function
            the loss function based on MSELoss to calculate the loss of VAE
        Returns
        -------
        vae_loss : torch.Tensor of shape [1]
            the loss of the VAE.
        dloss : torch.Tensor of shape [1]
            the loss of the discriminator.
        """

        config = self.config

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        if optimizer_idx == 0:
            ############
            #   VAE
            ############

            reconstructed_data_logit, z, mu, logVar = self.vae(
                dataset_batch, output_all=True
            )

            vae_rec_loss = self.criterion_reconstruct(
                reconstructed_data_logit, dataset_batch
            )

            # add KL loss
            KL = self.vae.calculate_log_prob_loss(z, mu, logVar) * config.kl_coef
            vae_rec_loss += KL

            # output of the vae should fool discriminator
            vae_out_d1, fc1 = self.discriminator(
                F.tanh(reconstructed_data_logit), output_all=True
            )
            vae_d_loss1 = self.criterion_label(vae_out_d1, real_label)
            ##### generate fake trees
            latent_size = self.vae.decoder_z_size
            # latent noise vector
            z = torch.randn(batch_size, latent_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.vae.generate_sample(z))
            # output of the vae should fool discriminator
            vae_out_d2, fc2 = self.discriminator(tree_fake, output_all=True)
            vae_d_loss2 = self.criterion_label(vae_out_d2, real_label)

            vae_d_loss = vae_d_loss1 + vae_d_loss2

            vae_d_loss = vae_d_loss * config.d_rec_coef

            _, fc_real = self.discriminator(dataset_batch, output_all=True)
            fc_real = fc_real.detach()
            # count similar to rec_loss
            FM_rec = config.FMrec_coef * self.criterion_FM(fc1, fc_real)
            # FM for fake data (and rescale with batch_size due to mean(fc))
            FM_gan = config.FMgan_coef * self.criterion_FM(
                torch.sum(fc2, 0), torch.sum(fc_real, 0)
            )

            vae_loss = vae_rec_loss + vae_d_loss + FM_rec + FM_gan
            self.record_loss(vae_rec_loss.detach().cpu().numpy(), loss_name="rec_loss")
            self.record_loss(KL.detach().cpu().numpy(), loss_name="KL_loss")
            self.record_loss(vae_d_loss.detach().cpu().numpy(), loss_name="vae_d_loss")
            self.record_loss(FM_rec.detach().cpu().numpy(), loss_name="FM_rec")
            self.record_loss(FM_gan.detach().cpu().numpy(), loss_name="FM_gan")

            return vae_loss

        if optimizer_idx == 1:

            ############
            #   discriminator
            ############

            # generate fake trees
            latent_size = self.vae.decoder_z_size
            # latent noise vector
            z = torch.randn(batch_size, latent_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.vae.generate_sample(z))

            # fake data (data from generator)
            # detach so no update to generator
            dout_fake = self.discriminator(tree_fake.clone().detach())
            dloss_fake = self.criterion_label(dout_fake, fake_label)
            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)
            dloss_real = self.criterion_label(dout_real, real_label)

            dloss = (dloss_fake + dloss_real) / 2  # scale the loss to one

            return dloss

    def generate_tree(self, num_trees=1):
        """
        the function to generate tree
        this function specifies the generator module of this model and pass to the parent generate_tree()
            see BaseModel generate_tree()
        Parameters
        ----------
        num_trees : int
            the number of trees to generate
        Returns
        -------
        result : numpy.ndarray shape [num_trees, res, res, res]
            the generated samples
        """
        generator = self.vae.vae_decoder
        return super().generate_tree(generator, num_trees=num_trees)


class GAN(BaseModel):
    """
    GAN
    This model contains a generator and a discriminator
    a random noise latent vector will be geenerated as the input of the model, and the generator will
        take the latent vector to the construct meshes. Then, the discriminator
        will give the score of the generated meshes on how close it is to real data
    The generator part of the model will train on how good the generated data fool the discriminator
    The discriminator part will train on prediction loss on classifying generated data and real data
    Attributes
    ----------
    config : Namespace
        dictionary of training parameters
    config.gen_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the Generator
    config.dis_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the Discriminator
    config.label_loss : string
        label_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assuming input to be logit (before sigmoid/tanh)
        this is the prediction loss for discriminator and generator
    config.g_lr : float
        the learning_rate of the Generator
    config.d_lr : float
        the learning_rate of the Discriminator
    config.FMgan_coef : float
        the coefficient of the Discriminator Feature Matching between real and generated
        This FM match the mean feature over real object to the mean feature over reconstructed object.
        This mean feature matching works to imporve GAN performance.
    config.gen_num_layer_unit : int/list
        the num_layer_unit for BaseModel setup_Generator()
        see also Generator class, BaseModel setup_Generator()
    config.dis_num_layer_unit : int/list
        the num_layer_unit for BaseModel setup_Discriminator()
        see also Discriminator class, BaseModel setup_Discriminator()
    self.generator : nn.Module
        the model component from setup_Generator()
    self.discriminator : nn.Module
        the model component from setup_Discriminator()
    self.criterion_label : nn Loss function
        the loss function based on config.label_loss
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """
        config = super(GAN, GAN).setup_config_arguments(config)

        # for config argument, see model description above
        default_model_config = {
            "gen_opt": "Adam",
            "dis_opt": "Adam",
            "label_loss": "BCELoss",
            "g_lr": 1e-5,
            "d_lr": 1e-5,
            "FMgan_coef": 0,
            "gen_num_layer_unit": [128, 256, 512, 256, 128, 128],
            "dis_num_layer_unit": [128, 128, 128, 256, 256, 128],
        }

        default_model_config.update(vars(config))
        config_rev = Namespace(**default_model_config)
        return config_rev

    def __init__(self, config):
        super(GAN, self).__init__(config)
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config
        self.save_hyperparameters("config")

        kernel_size = multiple_components_param(config.kernel_size, 2)
        fc_size = multiple_components_param(config.fc_size, 2)

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type
        self.generator = self.setup_Generator(
            "generator",
            num_layer_unit=config.gen_num_layer_unit,
            optimizer_option=config.gen_opt,
            learning_rate=config.g_lr,
            kernel_size=kernel_size[0],
            fc_size=fc_size[0],
        )
        self.discriminator = self.setup_Discriminator(
            "discriminator",
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
            kernel_size=kernel_size[1],
            fc_size=fc_size[1],
        )

        # loss function
        self.criterion_label = self.get_loss_function_with_logit(config.label_loss)
        self.criterion_FM = self.get_loss_function_with_logit("MSELoss")

    def forward(self, x):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input latent noise tensor in shape (B, Z)
            B = config.batch_size, Z = config.z_size
        Returns
        -------
        x : torch.Tensor of shape (B, 1)
            the discriminator score/logit of the generated data from the generator
                to show how close the generated data is looking like real data
        """
        x = self.generator(x)
        x = F.tanh(x)
        x = self.discriminator(x)
        return x

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
            here self.generator=0, self.discriminator=1
        self.criterion_label : nn Loss function
            the loss function based on config.label_loss to calculate the loss of generator/discriminator
        Returns
        -------
        gloss : torch.Tensor of shape [1]
            the loss of the generator.
        dloss : torch.Tensor of shape [1]
            the loss of the discriminator.
        """
        config = self.config

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        if optimizer_idx == 0:
            ############
            #   generator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))

            # tree_fake is already computed above
            dout_fake, fc1 = self.discriminator(tree_fake, output_all=True)
            # generator should generate trees that discriminator think they are real
            gloss = self.criterion_label(dout_fake, real_label)

            # Feature Matching
            _, fc_real = self.discriminator(dataset_batch, output_all=True)
            fc_real = fc_real.detach()
            # FM for fake data (and rescale with batch_size due to mean(fc))
            FM_gan = config.FMgan_coef * self.criterion_FM(
                torch.sum(fc1, 0), torch.sum(fc_real, 0)
            )
            self.record_loss(FM_gan.detach().cpu().numpy(), loss_name="FM_gan")

            return gloss + FM_gan

        if optimizer_idx == 1:

            ############
            #   discriminator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))

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

            return dloss

    def generate_tree(self, num_trees=1):
        """
        the function to generate tree
        this function specifies the generator module of this model and pass to the parent generate_tree()
            see BaseModel generate_tree()
        Parameters
        ----------
        num_trees : int
            the number of trees to generate
        Returns
        -------
        result : numpy.ndarray shape [num_trees, res, res, res]
            the generated samples
        """
        generator = self.generator
        return super().generate_tree(generator, num_trees=num_trees)


class GAN_Wloss(GAN):
    """
    GAN with Wasserstein loss
    This model is similar to the GAN model, but using Wasserstein loss to train
    reference: https://developers.google.com/machine-learning/gan/loss
    'Discriminator training just tries to make the output bigger for real instances than for fake instances,
    WGAN discriminator is actually called a "critic" instead of a "discriminator".'
    Attributes
    ----------
    config : Namespace
        dictionary of training parameters
    config.clip_value: float
        to make Wasserstein loss work, restriction of parameter values is needed
        for c = config.clip_value, the optimizer clip all parameters in the model to be in [-c,c]
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """
        config = super(GAN_Wloss, GAN_Wloss).setup_config_arguments(config)

        # for config argument, see model description above
        default_model_config = {
            "clip_value": 0.01,
        }

        default_model_config.update(vars(config))
        config_rev = Namespace(**default_model_config)
        return config_rev

    def configure_optimizers(self):
        """
        default function for pl.LightningModule to setup optimizer
        will be called automatically in pl.Trainer.fit()
        The optimizers for WGAN need to add restriction of parameter values
        for c = config.clip_value, the optimizer clip all parameters in the model to be in [-c,c]
        Parameters
        ----------
        config.clip_value : float
            clip all parameters in the model to be in [-c,c]
        Returns
        -------
        optimizer_list : list of torch.optim optimizer
            the optimizers of all model componenets
            See also BaseModel configure_optimizers()
        scheduler_list : list of torch.optim.lr_scheduler
            the cyclicLR schedulers of all model componenets if self.config.cyclicLR_magnitude attribute exists
            See also BaseModel configure_optimizers()
        """
        config = self.config
        discriminator = self.discriminator

        # clip critic (discriminator) gradient
        # no clip when gp is applied

        clip_value = config.clip_value
        for p in discriminator.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return super().configure_optimizers()

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
            here self.generator=0, self.discriminator=1
        Returns
        -------
        gloss : torch.Tensor of shape [1]
            the loss of the generator.
        dloss : torch.Tensor of shape [1]
            the loss of the discriminator.
        """
        config = self.config
        batch_size = dataset_batch.shape[0]

        # label no used in WGAN
        if optimizer_idx == 0:
            ############
            #   generator
            ############
            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))

            # tree_fake is already computed above
            dout_fake, fc1 = self.discriminator(tree_fake, output_all=True)

            # generator should maximize dout_fake
            gloss = -dout_fake.mean()

            # Feature Matching
            _, fc_real = self.discriminator(dataset_batch, output_all=True)
            fc_real = fc_real.detach()
            # FM for fake data (and rescale with batch_size due to mean(fc))
            FM_gan = config.FMgan_coef * self.criterion_FM(
                torch.sum(fc1, 0), torch.sum(fc_real, 0)
            )
            self.record_loss(FM_gan.detach().cpu().numpy(), loss_name="FM_gan")
            return gloss + FM_gan

        if optimizer_idx == 1:

            ############
            #   discriminator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)

            # fake data (data from generator)
            dout_fake = self.discriminator(
                tree_fake.clone().detach()
            )  # detach so no update to generator

            # d should maximize diff of real vs fake (dout_real - dout_fake)
            dloss = dout_fake.mean() - dout_real.mean()
            return dloss


class GAN_Wloss_GP(GAN):
    """
    GAN with Wasserstein loss with gradient penalty
    This model is similar to the GAN model, but using Wasserstein loss to train
    WGAN_GP use gradient penalty instead of clip value
    reference: https://developers.google.com/machine-learning/gan/loss
    'Discriminator training just tries to make the output bigger for real instances than for fake instances,
    WGAN discriminator is actually called a "critic" instead of a "discriminator".'
    Attributes
    ----------
    config : Namespace
        dictionary of training parameters
    config.gp_epsilon: float
        to make Wasserstein loss work, restriction of parameter values is needed
        gp_epsilon determines the scale of the calculated gradient penalty is used as the training loss
        gp_loss = gp_epsilon * gradient_penalty
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """
        config = super(GAN_Wloss_GP, GAN_Wloss_GP).setup_config_arguments(config)

        # for config argument, see model description above
        default_model_config = {
            "gp_epsilon": 2.0,
        }

        default_model_config.update(vars(config))
        config_rev = Namespace(**default_model_config)
        return config_rev

    @staticmethod
    def gradient_penalty(discriminator, gp_epsilon, real_tree, generated_tree):
        """
        GP for gradient_penalty
        WGAN_GP use gradient penalty to add model parameter restriction instead of clipping values
        calculate gradient penalty based on real sample and generated sample
        Parameters
        ----------
        real_tree : torch.Tensor
            the data from dataset/datamodule
        generated_tree : torch.Tensor
            the data generated from generator
        """
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
        prob_interpolated = discriminator(interpolated)

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
        return gp_epsilon * ((grad_norm - 1) ** 2).mean()

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
            here self.generator=0, self.discriminator=1
        Returns
        -------
        gloss : torch.Tensor of shape [1]
            the loss of the generator.
        dloss : torch.Tensor of shape [1]
            the loss of the discriminator.
        """
        config = self.config
        batch_size = dataset_batch.shape[0]

        # label no used in Wloss
        if optimizer_idx == 0:
            ############
            #   generator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))

            # tree_fake is already computed above
            dout_fake, fc1 = self.discriminator(tree_fake, output_all=True)

            # generator should maximize dout_fake
            gloss = -dout_fake.mean()

            # Feature Matching
            _, fc_real = self.discriminator(dataset_batch, output_all=True)
            fc_real = fc_real.detach()
            # FM for fake data (and rescale with batch_size due to mean(fc))
            FM_gan = config.FMgan_coef * self.criterion_FM(
                torch.sum(fc1, 0), torch.sum(fc_real, 0)
            )
            self.record_loss(FM_gan.detach().cpu().numpy(), loss_name="FM_gan")

            return gloss + FM_gan

        if optimizer_idx == 1:

            ############
            #   discriminator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.generator(z))

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)

            # fake data (data from generator)
            # detach so no update to generator
            dout_fake = self.discriminator(tree_fake.clone().detach())

            gp = self.gradient_penalty(
                self.discriminator, self.config.gp_epsilon, dataset_batch, tree_fake
            )
            # d should maximize diff of real vs fake (dout_real - dout_fake)
            dloss = dout_fake.mean() - dout_real.mean() + gp
            return dloss


#####
#   conditional models for training
#####
class CGAN(GAN):
    """
    Conditional GAN
    This model contains a generator, a discriminator, and a classifier
    a random noise latent vector will be geenerated as the input of the model, and the generator will
        take the latent vector to the construct meshes. Then, the discriminator
        will give the score of the generated meshes on how close it is to real data, and the
        classifier will give the score of generated meshes on how close it is to each class
    The generator part of the model will train on how good the generated data fool the discriminator
    The discriminator part will train on prediction loss on classifying generated data and real data
    The classifier part will train on classification loss on classifying generated data and real data to each class
    See also GAN for the attributes of generator and discriminator
    Attributes
    ----------
    config : Namespace
        dictionary of training parameters
    config.class_loss : string
        class_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assuming input to be logit (before sigmoid/tanh)
        this is the classification loss for classifier
    config.num_classes : int
        the number of classes in the dataset/datamodule
        if the datamodule is DataModule_process class, the config.num_classes there should be the same
    self.generator : nn.Module
        the model component from setup_Generator()
    self.discriminator : nn.Module
        the model component from setup_Discriminator() as discriminator
    self.classifier : nn.Module
        the model component from setup_Discriminator() as classifier
    self.criterion_label : nn Loss function
        the loss function based on config.label_loss
    self.criterion_class : nn Loss function
        the loss function based on config.class_loss
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """
        config = super(CGAN, CGAN).setup_config_arguments(config)

        # for config argument, see model description above
        default_model_config = {"num_classes": 10, "class_loss": "CrossEntropyLoss"}

        default_model_config.update(vars(config))
        config_rev = Namespace(**default_model_config)
        return config_rev

    def __init__(self, config):
        super(GAN, self).__init__(config)
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config
        self.save_hyperparameters("config")

        kernel_size = multiple_components_param(config.kernel_size, 3)
        fc_size = multiple_components_param(config.fc_size, 3)

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type
        self.generator = self.setup_Generator(
            "generator",
            num_layer_unit=config.gen_num_layer_unit,
            optimizer_option=config.gen_opt,
            learning_rate=config.g_lr,
            num_classes=config.num_classes,
            kernel_size=kernel_size[0],
            fc_size=fc_size[0],
        )
        self.discriminator = self.setup_Discriminator(
            "discriminator",
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
            kernel_size=kernel_size[1],
            fc_size=fc_size[1],
        )
        self.classifier = self.setup_Discriminator(
            "classifier",
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
            output_size=config.num_classes,
            kernel_size=kernel_size[2],
            fc_size=fc_size[2],
        )

        # add embedding layer for classes
        self.embedding = nn.Embedding(config.num_classes, config.num_classes)

        # loss function
        self.criterion_label = self.get_loss_function_with_logit(config.label_loss)
        self.criterion_class = self.get_loss_function_with_logit(config.class_loss)
        self.criterion_FM = self.get_loss_function_with_logit("MSELoss")

    def forward(self, x, c):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input latent noise tensor in shape (B, Z)
            B = config.batch_size, Z = config.z_size
        c : torch.Tensor
            the input class tensor in shape (B,)
                each element is the class index of the input x
        Returns
        -------
        d_predict : torch.Tensor of shape (B, 1)
            the discriminator score/logit of the generated data from the generator
                to show how close the generated data is looking like real data
        c_predict : torch.Tensor of shape (B, C)
            the classifier score/logit of the generated data from the generator
                to show how close the generated data is looking like the classes
                C = config.num_classes
        """
        # combine x and c into z
        z = self.merge_latent_and_class_vector(
            x, c, self.config.num_classes, embedding_fn=self.embedding
        )

        # classifier and discriminator
        x = self.generator(z)
        x = F.tanh(x)
        d_predict = self.discriminator(x)
        c_predict = self.classifier(x)
        return d_predict, c_predict

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
            here self.generator=0, self.discriminator=1, self.classifier=2
        self.criterion_label : nn Loss function
            the loss function based on config.label_loss to calculate the loss of generator/discriminator
        self.criterion_class : nn Loss function
            the loss function based on config.class_loss to calculate the loss of classifier
        Returns
        -------
        gloss : torch.Tensor of shape [1]
            the loss of the generator.
        dloss : torch.Tensor of shape [1]
            the loss of the discriminator.
        closs : torch.Tensor of shape [1]
            the loss of the classifier.
        """
        config = self.config

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
            z = self.merge_latent_and_class_vector(
                z, c_fake, self.config.num_classes, embedding_fn=self.embedding
            )

            tree_fake = F.tanh(self.generator(z))

            # tree_fake on Dis
            dout_fake, fc1 = self.discriminator(tree_fake, output_all=True)
            # generator should generate trees that discriminator think they are real
            gloss_d = self.criterion_label(dout_fake, real_label)

            # tree_fake on Cla
            cout_fake, cfc1 = self.classifier(tree_fake, output_all=True)
            gloss_c = self.criterion_class(cout_fake, c_fake)

            # Feature Matching
            _, fc_real = self.discriminator(dataset_batch, output_all=True)
            # _, cfc_real = self.classifier(dataset_batch, output_all=True)
            fc_real = fc_real.detach()
            # cfc_real = cfc_real.detach()
            # FM for fake data (and rescale with batch_size due to mean(fc))
            FM_gan1 = self.criterion_FM(torch.sum(fc1, 0), torch.sum(fc_real, 0))
            # FM_gan2 = self.criterion_FM(torch.mean(cfc1, 0), torch.mean(cfc_real, 0))
            FM_gan = config.FMgan_coef * FM_gan1
            self.record_loss(FM_gan.detach().cpu().numpy(), loss_name="FM_gan")

            gloss = gloss_d + gloss_c + FM_gan

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
            z = self.merge_latent_and_class_vector(
                z, c, self.config.num_classes, embedding_fn=self.embedding
            )

            # detach so no update to generator
            tree_fake = F.tanh(self.generator(z)).clone().detach()

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)
            dloss_real = self.criterion_label(dout_real, real_label)

            # fake data (data from generator)
            dout_fake = self.discriminator(tree_fake)
            dloss_fake = self.criterion_label(dout_fake, fake_label)

            # loss function (discriminator classify real data vs generated data)
            dloss = (dloss_real + dloss_fake) / 2

            return dloss

        if optimizer_idx == 2:

            ############
            #   classifier
            ############

            # # 128-d noise vector
            # z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            # # class vector
            # c_fake = (
            #     torch.randint(0, config.num_classes, (batch_size,))
            #     .type_as(dataset_batch)
            #     .to(torch.int64)
            # )

            # # combine z and c
            # z = self.merge_latent_and_class_vector(z, c_fake, self.config.num_classes, embedding_fn=self.embedding)

            # # detach so no update to generator
            # tree_fake = F.tanh(self.generator(z)).clone().detach()

            # # fake data (data from generator)
            # cout_fake = self.classifier(tree_fake)
            # closs_fake = self.criterion_class(cout_fake, c_fake)

            # real data (data from dataloader)
            cout_real = self.classifier(dataset_batch)
            closs_real = self.criterion_class(cout_real, dataset_indices)

            # loss function (discriminator classify real data vs generated data)
            # closs = (closs_real + closs_fake) / 2
            return closs_real

    def generate_tree(self, c, num_trees=1):
        """
        the function to generate tree
        this function specifies the generator module of this model and pass to the parent generate_tree()
            see BaseModel generate_tree()
        Parameters
        ----------
        num_trees : int
            the number of trees to generate
        c : int
            the class index of the class to generate
                the class index is based on the datamodule.class_list
                see datamodulePL.py datamodule_process class
        Returns
        -------
        result : numpy.ndarray shape [num_trees, res, res, res]
            the generated samples of the class with class index c
        """
        config = self.config
        generator = self.generator

        return super(GAN, self).generate_tree(
            generator,
            c=c,
            num_classes=config.num_classes,
            embedding_fn=self.embedding,
            num_trees=num_trees,
        )


class CVAEGAN(VAEGAN):
    """
    CVAE-GAN
    This model contains a vae, a discriminator, and a classifier
    the vae will take the input and class label to reconstruct meshes. Then, the discriminator
        will give the score of the generated meshes on how close it is to real data, and the
        classifier will give the score of generated meshes on how close it is to each class
    The vae part of the model will train on how good the reconstructed data fool the discriminator,
        match the class_label, and how the reconstructed data looks like the input data
    The discriminator part will train on prediction loss on classifying generated data and real data
    The classifier part will train on classification loss on classifying generated data and real data to each class
    See also GAN for the attributes of generator and discriminator
    Attributes
    ----------
    config : Namespace
        dictionary of training parameters
    config.class_loss : string
        class_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assuming input to be logit (before sigmoid/tanh)
        this is the classification loss for classifier
    config.num_classes : int
        the number of classes in the dataset/datamodule
        if the datamodule is DataModule_process class, the config.num_classes there should be the same
    config.c_rec_coef : float
        the coefficient of the Classifier loss compared to the reconstruction loss
    self.vae : nn.Module
        the model component from setup_VAE()
    self.discriminator : nn.Module
        the model component from setup_Discriminator() as discriminator
    self.classifier : nn.Module
        the model component from setup_Discriminator() as classifier
    self.criterion_label : nn Loss function
        the loss function based on config.label_loss
    self.criterion_class : nn Loss function
        the loss function based on config.class_loss
    self.criterion_reconstruct : nn Loss function
        the loss function based on MSELoss
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """
        config = super(CVAEGAN, CVAEGAN).setup_config_arguments(config)

        # for config argument, see model description above
        default_model_config = {
            "num_classes": 10,
            "class_loss": "CrossEntropyLoss",
            "c_rec_coef": 1.0,
        }

        default_model_config.update(vars(config))
        config_rev = Namespace(**default_model_config)
        return config_rev

    def __init__(self, config):
        super(VAEGAN, self).__init__(config)
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config
        self.save_hyperparameters("config")

        kernel_size = multiple_components_param(config.kernel_size, 4)
        fc_size = multiple_components_param(config.fc_size, 4)

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type

        self.vae = self.setup_VAE(
            model_name="VAE",
            encoder_num_layer_unit=config.encoder_num_layer_unit,
            decoder_num_layer_unit=config.decoder_num_layer_unit,
            optimizer_option=config.vae_opt,
            learning_rate=config.vae_lr,
            num_classes=config.num_classes,
            kernel_size=kernel_size[:2],
            fc_size=fc_size[:2],
        )
        self.discriminator = self.setup_Discriminator(
            "discriminator",
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
            kernel_size=kernel_size[2],
            fc_size=fc_size[2],
        )
        self.classifier = self.setup_Discriminator(
            "classifier",
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
            output_size=config.num_classes,
            kernel_size=kernel_size[3],
            fc_size=fc_size[3],
        )

        # loss function
        self.criterion_label = self.get_loss_function_with_logit(config.label_loss)
        self.criterion_class = self.get_loss_function_with_logit(config.class_loss)
        # rec loss function is modified to handle pos_weight
        self.criterion_reconstruct = (
            lambda gen, data: torch.sum(
                nn.MSELoss(reduction="none")(F.tanh(gen), data)
                * self.calculate_pos_weight(data)
            )
            / data.shape[0]
        )
        self.criterion_FM = self.get_loss_function_with_logit("MSELoss")

    def forward(self, x, c):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input latent noise tensor in shape (B, Z)
            B = config.batch_size, Z = config.z_size
        c : torch.Tensor
            the input class tensor in shape (B,)
                each element is the class index of the input x
        Returns
        -------
        d_predict : torch.Tensor of shape (B, 1)
            the discriminator score/logit of the generated data from the generator
                to show how close the generated data is looking like real data
        c_predict : torch.Tensor of shape (B, C)
            the classifier score/logit of the generated data from the generator
                to show how close the generated data is looking like the classes
                C = config.num_classes
        """
        x = self.vae(x, c)
        x = F.tanh(x)
        d_predict = self.discriminator(x)
        c_predict = self.classifier(x)
        return d_predict, c_predict

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
            here self.generator=0, self.discriminator=1, self.classifier=2
        self.criterion_label : nn Loss function
            the loss function based on config.label_loss to calculate the loss of generator/discriminator
        self.criterion_class : nn Loss function
            the loss function based on config.class_loss to calculate the loss of classifier
        self.criterion_reconstruct : nn Loss function
            the loss function based on MSELoss to calculate the loss of VAE
        Returns
        -------
        vae_loss : torch.Tensor of shape [1]
            the loss of the vae.
        dloss : torch.Tensor of shape [1]
            the loss of the discriminator.
        closs : torch.Tensor of shape [1]
            the loss of the classifier.
        """
        config = self.config

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        if optimizer_idx == 0:
            ############
            #   generator
            ############

            reconstructed_data_logit, z, mu, logVar = self.vae(
                dataset_batch, dataset_indices, output_all=True
            )

            vae_rec_loss = self.criterion_reconstruct(
                reconstructed_data_logit, dataset_batch
            )
            self.record_loss(vae_rec_loss.detach().cpu().numpy(), loss_name="rec_loss")

            # add KL loss
            KL = self.vae.calculate_log_prob_loss(z, mu, logVar) * config.kl_coef
            self.record_loss(KL.detach().cpu().numpy(), loss_name="KL loss")
            vae_rec_loss += KL

            # output of the vae should fool discriminator
            vae_out_d1, fc1 = self.discriminator(
                F.tanh(reconstructed_data_logit), output_all=True
            )
            vae_d_loss1 = self.criterion_label(vae_out_d1, real_label)
            ##### generate fake trees
            # latent noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            z = self.merge_latent_and_class_vector(
                z,
                dataset_indices,
                self.config.num_classes,
                embedding_fn=self.vae.embedding,
            )
            tree_fake = F.tanh(self.vae.generate_sample(z))
            # output of the vae should fool discriminator
            vae_out_d2, fc2 = self.discriminator(tree_fake, output_all=True)
            vae_d_loss2 = self.criterion_label(vae_out_d2, real_label)
            vae_d_loss = (vae_d_loss1 + vae_d_loss2) / 2

            # tree_fake on Cla
            vae_out_c1, cfc1 = self.classifier(
                F.tanh(reconstructed_data_logit), output_all=True
            )
            vae_c_loss1 = self.criterion_class(vae_out_c1, dataset_indices)
            vae_out_c2, cfc2 = self.classifier(tree_fake, output_all=True)
            vae_c_loss2 = self.criterion_class(vae_out_c2, dataset_indices)
            vae_c_loss = (vae_c_loss1 + vae_c_loss2) / 2

            vae_d_loss = vae_d_loss * config.d_rec_coef
            self.record_loss(vae_d_loss.detach().cpu().numpy(), loss_name="vae_d_loss")
            vae_c_loss = vae_c_loss * config.c_rec_coef
            self.record_loss(vae_c_loss.detach().cpu().numpy(), loss_name="vae_c_loss")

            # Feature Matching
            _, fc_real = self.discriminator(dataset_batch, output_all=True)
            # _, cfc_real = self.classifier(dataset_batch, output_all=True)
            fc_real = fc_real.detach()
            # cfc_real = cfc_real.detach()
            # count similar to rec_loss
            FM_rec1 = config.FMrec_coef * self.criterion_FM(fc1, fc_real)
            # FM_rec2 = config.FMrec_coef * self.criterion_FM(cfc1, cfc_real)
            FM_rec = FM_rec1
            # FM for fake data (and rescale with batch_size due to mean(fc))
            FM_gan1 = self.criterion_FM(torch.sum(fc2, 0), torch.sum(fc_real, 0))
            # FM_gan2 = self.criterion_FM(torch.mean(cfc2, 0), torch.mean(cfc_real, 0))
            FM_gan = config.FMgan_coef * FM_gan1
            self.record_loss(FM_rec.detach().cpu().numpy(), loss_name="FM_rec")
            self.record_loss(FM_gan.detach().cpu().numpy(), loss_name="FM_gan")

            vae_loss = vae_rec_loss + vae_d_loss + vae_c_loss + FM_rec + FM_gan

            return vae_loss

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
            z = self.merge_latent_and_class_vector(
                z, c, self.config.num_classes, embedding_fn=self.vae.embedding
            )

            # detach so no update to generator
            tree_fake = F.tanh(self.vae.vae_decoder(z)).clone().detach()

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)
            dloss_real = self.criterion_label(dout_real, real_label)

            # fake data (data from generator)
            dout_fake = self.discriminator(tree_fake)
            dloss_fake = self.criterion_label(dout_fake, fake_label)

            # loss function (discriminator classify real data vs generated data)
            dloss = (dloss_real + dloss_fake) / 2

            return dloss

        if optimizer_idx == 2:

            ############
            #   classifier
            ############

            # reconstructed_data = self.vae(dataset_batch, dataset_indices)
            # reconstructed_data = F.tanh(reconstructed_data).clone().detach()

            # # fake data (data from generator)
            # cout_fake = self.classifier(reconstructed_data)
            # closs_fake = self.criterion_class(cout_fake, dataset_indices)

            # real data (data from dataloader)
            cout_real = self.classifier(dataset_batch)
            closs_real = self.criterion_class(cout_real, dataset_indices)

            # loss function (discriminator classify real data vs generated data)
            # closs = (closs_real + closs_fake) / 2
            return closs_real

    def generate_tree(self, c, num_trees=1):
        """
        the function to generate tree
        this function specifies the generator module of this model and pass to the parent generate_tree()
            see BaseModel generate_tree()
        Parameters
        ----------
        num_trees : int
            the number of trees to generate
        c : int
            the class index of the class to generate
                the class index is based on the datamodule.class_list
                see datamodulePL.py datamodule_process class
        Returns
        -------
        result : numpy.ndarray shape [num_trees, res, res, res]
            the generated samples of the class with class index c
        """
        config = self.config
        generator = self.vae.vae_decoder

        return super(VAEGAN, self).generate_tree(
            generator,
            c=c,
            num_classes=config.num_classes,
            embedding_fn=self.vae.embedding,
            num_trees=num_trees,
        )


# TODO ZVAEGAN description
class ZVAEGAN(VAEGAN):
    """
    ZVAE-GAN
    """

    @staticmethod
    def setup_config_arguments(config):
        """
        The default_model_config listed all the parameters for the model

        from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the default_model_config,
            and then add only the missing parameter values to the user config dict
        (the values in default_model_config will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()

        default_model_config containing default values for all necessary argument
            The arguments in the default_model_config will be added to config if missing.
            If config already have the arguments, the values won't be replaced.

        Parameters
        ----------
        config : Namespace
            This is the arguments passed by user, waiting for filling in missing arguments.

        Returns
        -------
        config_rev : Namespace
            the revised config with missing argument filled in.
        """
        config = super(ZVAEGAN, ZVAEGAN).setup_config_arguments(config)

        # for config argument, see model description above
        default_model_config = {
            "num_classes": 10,
            "class_std": -1,  # class_std for specifying N(class_embed, std) for training conditional data
        }

        default_model_config.update(vars(config))
        config_rev = Namespace(**default_model_config)
        return config_rev

    def __init__(self, config):
        super(VAEGAN, self).__init__(config)
        # add missing default parameters
        config = self.setup_config_arguments(config)
        self.config = config
        self.save_hyperparameters("config")

        kernel_size = multiple_components_param(config.kernel_size, 3)
        fc_size = multiple_components_param(config.fc_size, 3)
        class_std = config.class_std
        if class_std < 0:
            class_std = 1 - config.num_classes / (config.num_classes + config.z_size)

        # create components
        # set component as an attribute to the model
        # so PL can set tensor device type

        # modified VAE
        self.vae = self.setup_VAE_mod(
            model_name="VAE",
            encoder_num_layer_unit=config.encoder_num_layer_unit,
            decoder_num_layer_unit=config.decoder_num_layer_unit,
            optimizer_option=config.vae_opt,
            learning_rate=config.vae_lr,
            num_classes=config.num_classes,
            kernel_size=kernel_size[:2],
            fc_size=fc_size[:2],
            class_std=class_std,
        )

        self.discriminator = self.setup_Discriminator(
            "discriminator",
            num_layer_unit=config.dis_num_layer_unit,
            optimizer_option=config.dis_opt,
            learning_rate=config.d_lr,
            kernel_size=kernel_size[2],
            fc_size=fc_size[2],
        )

        # loss function
        self.criterion_label = self.get_loss_function_with_logit(config.label_loss)

        # rec loss function is modified to handle pos_weight
        self.criterion_reconstruct = (
            lambda gen, data: torch.sum(
                nn.MSELoss(reduction="none")(F.tanh(gen), data)
                * self.calculate_pos_weight(data)
            )
            / data.shape[0]
        )
        self.criterion_FM = self.get_loss_function_with_logit("MSELoss")

    def calculate_loss(self, dataset_batch, dataset_indices=None, optimizer_idx=0):
        """
        function to calculate loss of each of the model components
        the model_name of the component: self.model_name_list[optimizer_idx]
        Parameters
        ----------
        dataset_batch : torch.Tensor
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, 1, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        optimizer_idx : int
            the index of the optimizer
            the optimizer_idx is based on the setup order of model components
            the model_name of the component: self.model_name_list[optimizer_idx]
            here self.generator=0, self.discriminator=1, self.classifier=2
        self.criterion_label : nn Loss function
            the loss function based on config.label_loss to calculate the loss of generator/discriminator
        self.criterion_class : nn Loss function
            the loss function based on config.class_loss to calculate the loss of classifier
        self.criterion_reconstruct : nn Loss function
            the loss function based on MSELoss to calculate the loss of VAE
        Returns
        -------
        vae_loss : torch.Tensor of shape [1]
            the loss of the vae.
        dloss : torch.Tensor of shape [1]
            the loss of the discriminator.
        closs : torch.Tensor of shape [1]
            the loss of the classifier.
        """
        config = self.config

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        dataset_indices_cond_mask = (dataset_indices >= 0).type_as(dataset_batch)

        if optimizer_idx == 0:
            ############
            #   generator
            ############

            reconstructed_data_logit, z, mu, logVar = self.vae(
                dataset_batch, dataset_indices, output_all=True
            )
            vae_rec_loss = self.criterion_reconstruct(
                reconstructed_data_logit, dataset_batch
            )
            self.record_loss(vae_rec_loss.detach().cpu().numpy(), loss_name="rec_loss")

            # add class KL loss
            # mask uncond(-1) with 0, and then make target_mu 0
            masked_dataset_indices = (
                dataset_indices * dataset_indices_cond_mask.type_as(dataset_indices)
            )
            target_mu = self.vae.embedding(
                masked_dataset_indices
            ) * dataset_indices_cond_mask.reshape((-1, 1))
            class_KL = (
                self.vae.calculate_log_prob_loss(
                    z,
                    mu,
                    logVar,
                    target_mu=target_mu,
                    cond_mask=dataset_indices_cond_mask,
                )
                * config.kl_coef
            )
            self.record_loss(class_KL.detach().cpu().numpy(), loss_name="class KL loss")
            vae_rec_loss = vae_rec_loss + class_KL

            # output of the vae should fool discriminator
            vae_out_d1, fc1 = self.discriminator(
                F.tanh(reconstructed_data_logit), output_all=True
            )
            vae_d_loss1 = self.criterion_label(vae_out_d1, real_label)

            ##### generate fake trees
            # latent noise vector (conditional)
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake_cond = F.tanh(self.vae.generate_sample(z, dataset_indices))
            # output of the vae should fool discriminator
            vae_out_d2, fc2 = self.discriminator(tree_fake_cond, output_all=True)
            vae_d_loss2 = self.criterion_label(vae_out_d2, real_label)

            # latent noise vector (unconditional)
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            tree_fake_uncond = F.tanh(self.vae.generate_sample(z))
            # output of the vae should fool discriminator
            vae_out_d3, fc3 = self.discriminator(tree_fake_uncond, output_all=True)
            vae_d_loss3 = self.criterion_label(vae_out_d3, real_label)

            vae_d_loss = (vae_d_loss1 + vae_d_loss2 + vae_d_loss3) / 3
            vae_d_loss = vae_d_loss * config.d_rec_coef
            self.record_loss(vae_d_loss.detach().cpu().numpy(), loss_name="vae_d_loss")

            # Feature Matching
            _, fc_real = self.discriminator(dataset_batch, output_all=True)
            fc_real = fc_real.detach()
            # count similar to rec_loss
            FM_rec = config.FMrec_coef * self.criterion_FM(fc1, fc_real)
            # FM for fake data (and rescale with batch_size due to mean(fc))
            FM_gan1 = self.criterion_FM(torch.sum(fc2, 0), torch.sum(fc_real, 0))
            FM_gan2 = self.criterion_FM(torch.sum(fc3, 0), torch.sum(fc_real, 0))
            FM_gan = config.FMgan_coef * (FM_gan1 + FM_gan2)
            self.record_loss(FM_rec.detach().cpu().numpy(), loss_name="FM_rec")
            self.record_loss(FM_gan.detach().cpu().numpy(), loss_name="FM_gan")

            vae_loss = vae_rec_loss + vae_d_loss + FM_rec + FM_gan

            return vae_loss

        if optimizer_idx == 1:

            ############
            #   discriminator
            ############

            # 128-d noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)

            # detach so no update to generator
            tree_fake = F.tanh(self.vae.vae_decoder(z)).clone().detach()

            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)
            dloss_real = self.criterion_label(dout_real, real_label)

            # fake data (data from generator)
            dout_fake = self.discriminator(tree_fake)
            dloss_fake = self.criterion_label(dout_fake, fake_label)

            # loss function (discriminator classify real data vs generated data)
            dloss = (dloss_real + dloss_fake) / 2

            return dloss

    def generate_tree(self, c=None, num_trees=1):
        """
        the function to generate tree
        this function specifies the generator module of this model and pass to the parent generate_tree()
            see BaseModel generate_tree()
        Parameters
        ----------
        num_trees : int
            the number of trees to generate
        c : int
            the class index of the class to generate
                the class index is based on the datamodule.class_list
                see datamodulePL.py datamodule_process class
        Returns
        -------
        result : numpy.ndarray shape [num_trees, res, res, res]
            the generated samples of the class with class index c
        """
        config = self.config
        generator = self.vae.vae_decoder

        batch_size = self.config.batch_size
        resolution = generator.resolution

        if batch_size > num_trees:
            batch_size = num_trees

        result = None

        num_runs = int(np.ceil(num_trees / batch_size))
        # ignore discriminator
        for i in range(num_runs):
            z = torch.randn(batch_size, self.config.z_size).type_as(
                generator.gen_fc.weight
            )

            if c is not None:
                # turn class vector the same device as z, but with dtype Long
                c = torch.ones(batch_size) * c
                c = c.type_as(z).to(torch.int64)
                tree_fake = self.vae.generate_sample(z, c)
            else:
                tree_fake = self.vae.generate_sample(z)

            selected_trees = tree_fake[:, 0, :, :, :].detach().cpu().numpy()
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
#   building blocks for models
#####


class VAE(nn.Module):
    """
    The VAE class
    This class takes a generator as the decoder and a discriminator as the encoder
    if the VAE is unconditional, the latent vector of the decoder
        = noise reparameterization of the output of the encoder
    if the VAE is conditional, the latent vector of the decoder
        = noise reparameterization of the output of the encoder + one-hot class vector
    reference: https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner/blob/master/CVAE-GAN/CVAE-GAN.py
    reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
    reference: https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/cvae.ipynb
    Attributes
    ----------
    encoder : nn.Module
        the nn module to be the encoder, assume to be in discriminator class
    decoder : nn.Module
        the nn module to be the encoder, assume to be in generator class
    num_classes : int/None
        if the data is conditional, the latent vector of the decoder should be
            the output vector of the encoder + one-hot class vector
    """

    def __init__(self, encoder, decoder, num_classes=None, dim_class_embedding=64):
        super(VAE, self).__init__()
        assert encoder.resolution == decoder.resolution
        self.resolution = decoder.resolution
        self.encoder_z_size = encoder.output_size
        self.decoder_z_size = decoder.z_size
        self.num_classes = num_classes
        if num_classes is not None:
            assert (self.decoder_z_size - self.encoder_z_size) == self.num_classes
        # VAE
        self.vae_encoder = encoder

        self.encoder_mean = nn.Linear(self.encoder_z_size, self.encoder_z_size)
        self.encoder_logvar = nn.Linear(self.encoder_z_size, self.encoder_z_size)

        self.vae_decoder = decoder
        if num_classes is None:
            self.embedding = None
        else:
            self.embedding = nn.Embedding(num_classes, dim_class_embedding)

    def noise_reparameterize(self, mean, logvar):
        """
        noise reparameterization of the VAE
        reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        Parameters
        ----------
        mean : torch.Tensor of shape (B, Z)
        logvar : torch.Tensor of shape (B, Z)
        """

        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z

    def calculate_log_prob_loss(self, z, mu, logVar):
        """
        calculate log_prob loss (KL/ELBO??) for VAE based on the mean and logvar used for noise_reparameterize()
        reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        See VAE class
        Parameters
        ----------
        mu : torch.Tensor
        logVar : torch.Tensor
        """

        # reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        std = torch.exp(logVar / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        kl = log_qz - log_pz
        loss = kl.mean()
        return loss

    def forward(self, x, c=None, output_all=False):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input data tensor in shape (B, 1, R, R, R)
                B = config.batch_size, R = encoder.input_size
        c : torch.Tensor
            the input class tensor in shape (B,)
                each element is the class index of the input x
        output_all : boolean
            whether to also output x_mean and x_logvar
        Returns
        -------
        x : torch.Tensor of shape (B, 1, R, R, R)
            the reconstructed data from the VAE based on input x
        x_mean : torch.Tensor of shape (B, Z)
            the mean for noise_reparameterization
        x_logvar : torch.Tensor of shape (B, Z)
            the logvar for noise_reparameterization
        """

        # VAE
        f = self.vae_encoder(x)

        x_mean = self.encoder_mean(f)
        x_logvar = self.encoder_logvar(f)

        z = self.noise_reparameterize(x_mean, x_logvar)

        # handle class vector
        if c is not None:
            x = BaseModel.merge_latent_and_class_vector(
                z, c, self.num_classes, embedding_fn=self.embedding
            )
        else:
            x = z

        x = self.vae_decoder(x)
        if output_all:
            return x, z, x_mean, x_logvar
        else:
            return x

    def generate_sample(self, z, c=None):
        """
        the function to generate sample given a latent vector (and class index)
        Parameters
        ----------
        z : torch.Tensor
            the input latent noise tensor in shape (B, Z)
                B = config.batch_size, Z = decoder.z_size
        c : torch.Tensor
            the input class tensor in shape (B,)
                each element is the class index of the input x
        Returns
        -------
        x : torch.Tensor
            the generated data in shape (B, 1, R, R, R) from the decoder based on latent vector x
                B = config.batch_size, R = decoder.output_size
        """
        # handle class vector
        if c is not None:
            z = BaseModel.merge_latent_and_class_vector(
                z, c, self.num_classes, embedding_fn=self.embedding
            )

        x = self.vae_decoder(z)
        return x


# TODO: VAE_mod doc
class VAE_mod(nn.Module):
    def __init__(self, encoder, decoder, num_classes=None, class_std=1):
        super(VAE_mod, self).__init__()
        if num_classes:
            self.embedding = nn.Embedding(num_classes, decoder.z_size)
        else:
            self.embedding = None

        self.resolution = decoder.resolution
        self.encoder_z_size = encoder.output_size
        self.decoder_z_size = decoder.z_size

        # VAE
        self.vae_encoder = encoder

        self.encoder_mean = nn.Linear(self.encoder_z_size, self.encoder_z_size)
        self.encoder_logvar = nn.Linear(self.encoder_z_size, self.encoder_z_size)

        self.vae_decoder = decoder
        self.num_classes = num_classes
        self.class_std = class_std

    def noise_reparameterize(self, mean, logvar):
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z

    def calculate_log_prob_loss(self, z, mu, logVar, target_mu=None, cond_mask=None):
        """
        calculate log_prob loss (KL/ELBO??) for VAE based on the mean and logvar used for noise_reparameterize()
        reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        See VAE class
        Parameters
        ----------
        mu : torch.Tensor
        logVar : torch.Tensor
        """

        # add target mu and std for VAE_mod
        if target_mu is None:
            target_mu = torch.zeros_like(mu)
        if cond_mask is None:
            target_std = torch.ones_like(logVar)
        else:
            cond_mask = cond_mask.type_as(logVar)
            target_std = self.class_std * cond_mask + 1 * (1 - cond_mask)
            target_std = torch.ones_like(logVar) * target_std.reshape((-1, 1))

        # reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        std = torch.exp(logVar / 2)
        p = torch.distributions.Normal(target_mu, target_std)
        q = torch.distributions.Normal(mu, std)

        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        kl = log_qz - log_pz
        loss = kl.mean()

        # also constraint target_mu (class_vector to be in N(0,1))
        # maximize log prob of target_mu from N(0,1)
        n = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        log_nc = n.log_prob(target_mu)
        loss_c = (-log_nc).mean()
        return loss + loss_c

    def forward(self, x, c=None, output_all=False):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input data tensor in shape (B, 1, R, R, R)
                B = config.batch_size, R = encoder.input_size
        c : torch.Tensor
            the input class tensor in shape (B,)
                each element is the class index of the input x
        output_all : boolean
            whether to also output x_mean and x_logvar
        Returns
        -------
        x : torch.Tensor of shape (B, 1, R, R, R)
            the reconstructed data from the VAE based on input x
        x_mean : torch.Tensor of shape (B, Z)
            the mean for noise_reparameterization
        x_logvar : torch.Tensor of shape (B, Z)
            the logvar for noise_reparameterization
        """

        # VAE
        f = self.vae_encoder(x)

        x_mean = self.encoder_mean(f)
        x_logvar = self.encoder_logvar(f)

        z = self.noise_reparameterize(x_mean, x_logvar)
        x = self.vae_decoder(z)
        if output_all:
            return x, z, x_mean, x_logvar
        else:
            return x

    def generate_sample(self, z, c=None):
        """
        the function to generate sample given a latent vector (and class index)
        Parameters
        ----------
        z : torch.Tensor
            the input latent noise tensor in shape (B, Z)
                B = config.batch_size, Z = decoder.z_size
        c : torch.Tensor
            the input class tensor in shape (B,)
                each element is the class index of the input x
        Returns
        -------
        x : torch.Tensor
            the generated data in shape (B, 1, R, R, R) from the decoder based on latent vector x
                B = config.batch_size, R = decoder.output_size
        """
        # handle class vector
        if c is not None:
            # find out uncond label and class label
            c_cond_mask = (c > 0).type_as(c)
            # replace uncond label with 0 for embedding conversion
            c = self.embedding(c * c_cond_mask).type_as(z)
            # use original z for uncond generation, use class_vector + class_std * z for conditional generation
            c_cond_mask = c_cond_mask.reshape((-1, 1))
            z = z * (1 - c_cond_mask) + c_cond_mask * (c + self.class_std * z)

        x = self.vae_decoder(z)
        return x


class Generator(nn.Module):
    """
    The generator class
    This class serves as the generator and VAE decoder of models above
    the model starts with (linear/convT) depending on fc_size, with several ConvTLayer, and ends with 1 conv layer.
    Each ConvTLayer of the model contains (dropout, convT, batchnorm, activations)

    Attributes
    ----------
    z_size : int
        the latent vector size of the model
        latent vector size determines the size of the generated input, and the size of the compressed latent vector in VAE
    resolution : int
        the output size of the generator
        given latent vector (B,Z), the output will be (B,1,R,R,R)
            Z = z_size, R = resolution, B = batch_size
    num_layer_unit : int/list
        the number of unit in the ConvT layer
            if int, every ConvT will have the same specified number of unit.
            if list, every ConvT layer between the upsampling layers will have the specified number of unit.
    dropout_prob : float
        the dropout probability of the generator models (see torch Dropout3D)
        the generator use ConvT-BatchNorm-activation-dropout structure
    activations : nn actication function
        the actication function used for all layers except the last layer of all models
    kernel_size : int
            the kernel size of the convT layer. padding will be adjusted so the output_size of the model is not affected
    fc_size : int
        the size of the input volume for the first convT layer.
        For fc_size=2, the last fc layer will output B*unit_list[0]*fc_size**3, and reshape into (B,unit_list[0],fc_size, fc_size, fc_size).
        The lower the value, the number of upsampling layer and convT layer increases
        (number of ConvTLayer = int(np.log2(self.resolution)) - int(np.log2(self.fc_size))).
    """

    def ConvTLayer(self, in_channel, out_channel):
        """
        helper function to construct (dropout, convT, batchnorm, activations)
        Parameters
        ----------
        in_channel : int
            the number of input channel for ConvTranspose3d() layer
        out_channel : int
            the number of out channel for ConvTranspose3d() layer
        self.dropout_prob : int
        self.activations : nn actication function
        self.kernel_size : int
            described in Generator Attributes above.
        Returns
        -------
        layer_modules : nn.Sequential(*layer_module)
            a block of layers with (dropout, convT, batchnorm, activations)
        """

        dropout_prob = self.dropout_prob
        activations = self.activations
        kernel_size = self.kernel_size

        layer_module = []
        # dropout
        if dropout_prob > 0:
            layer_module.append(nn.Dropout3d(dropout_prob))

        # convT layer
        if kernel_size == 3:
            layer_module.append(nn.Upsample(scale_factor=2, mode="trilinear"))
            conv_layer = nn.ConvTranspose3d(
                in_channel, out_channel, kernel_size=3, stride=1, padding=1
            )
        elif kernel_size == 4:
            conv_layer = nn.ConvTranspose3d(
                in_channel, out_channel, kernel_size=4, stride=2, padding=1
            )
        else:
            conv_layer = nn.ConvTranspose3d(
                in_channel,
                out_channel,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            )

        # normalization
        # batch norm
        layer_module.append(conv_layer)
        layer_module.append(nn.BatchNorm3d(out_channel))

        # activation
        layer_module.append(activations)

        return nn.Sequential(*layer_module)

    def __init__(
        self,
        z_size=128,
        resolution=64,
        num_layer_unit=32,
        kernel_size=3,
        fc_size=2,
        dropout_prob=0.0,
        activations=nn.LeakyReLU(0.01, True),
    ):
        super(Generator, self).__init__()

        self.z_size = z_size
        self.fc_size = fc_size
        self.resolution = resolution

        self.dropout_prob = dropout_prob
        self.activations = activations
        self.kernel_size = kernel_size

        # number of conv layer in the model
        # if fc_size=2, resolution=32, it should be 4 layers
        self.num_layers = int(np.log2(self.resolution)) - int(np.log2(self.fc_size))
        unit_list_size = self.num_layers + 1

        # check num_layer_unit list
        if type(num_layer_unit) is list:
            if len(num_layer_unit) < unit_list_size:
                message = f"For resolution={resolution}, fc_size={fc_size}, the list of num_layer_unit should have {unit_list_size} elements."
                raise Exception(message)
            if len(num_layer_unit) > unit_list_size:
                num_layer_unit = num_layer_unit[:unit_list_size]
                message = f"For resolution={resolution}, fc_size={fc_size}, the list of num_layer_unit should have {unit_list_size} elements. Trimming num_layer_unit to {num_layer_unit}"
                print(message)
            unit_list = num_layer_unit
        elif type(num_layer_unit) is int:
            unit_list = [num_layer_unit] * unit_list_size
        else:
            raise Exception("num_layer_unit should be int of list of int.")

        gen_fc_module = []
        gen_module = []
        if self.fc_size > 1:
            num_fc_units = unit_list[0] * self.fc_size * self.fc_size * self.fc_size
            gen_fc_module.append(nn.Linear(self.z_size, num_fc_units))
            gen_fc_module.append(activations)
        else:
            gen_module.append(
                nn.ConvTranspose3d(
                    self.z_size, unit_list[0], kernel_size=3, stride=1, padding=1
                )
            )

        # n layer conv/deconv
        for i in range(self.num_layers):
            gen_module.append(self.ConvTLayer(unit_list[i], unit_list[i + 1]))

        # #final layer
        gen_module.append(
            nn.Conv3d(unit_list[-1], 1, kernel_size=3, stride=1, padding=1)
        )

        self.gen_fc_module = nn.Sequential(*gen_fc_module)
        self.gen_module = nn.Sequential(*gen_module)

        # placeholder
        self.gen_fc = nn.Linear(1, 1)

    def forward(self, x):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input latent noise tensor in shape (B, Z)
            B = config.batch_size, Z = z_size
        Returns
        -------
        x : torch.Tensor
            the generated data in shape (B, 1, R, R, R) from the Generator based on latent vector x
            B = config.batch_size, R = resolution
        """
        if self.fc_size > 1:
            x = self.gen_fc_module(x)
        x = x.view(x.shape[0], -1, self.fc_size, self.fc_size, self.fc_size)
        x = self.gen_module(x)
        return x


class Discriminator(nn.Module):
    """
    The Discriminator class
    This class serves as the discriminator, the classifier, and VAE encoder of models above
    The model starts with 1 conv layer, with several ConvLayer, and ends (conv/linear) depends on fc_size.
    Each ConvLayer of the model contains (dropout, conv, batchnorm, activations)
    Attributes
    ----------
    output_size : int
        the output size of the discriminator
        the input will be in the shape (B,S)
            S = output_size, B = batch_size
    resolution : int
        the input size of the discriminator
        the input will be in the shape (B,1,R,R,R)
            R = resolution, B = batch_size
    num_layer_unit : int/list
        the number of unit in the ConvT layer
            if int, every ConvT will have the same specified number of unit.
            if list, every ConvT layer between the upsampling layers will have the specified number of unit.
    dropout_prob : float
        the dropout probability of the generator models (see torch Dropout3D)
        the generator use ConvT-BatchNorm-activation-dropout structure
    activations : nn actication function
        the actication function used for all layers except the last layer of all models
    kernel_size : int
        the kernel size of the conv layer. padding will be adjusted so the output_size of the model is not affected
    fc_size : int
        the size of the output volume for the last conv layer.
        For fc_size=2, the last conv layer will output (B,unit_list[-1],fc_size, fc_size, fc_size), and flatten that for fc_layer.
        The lower the value, the number of downsampling layer and conv layer increases
        (number of ConvLayer = int(np.log2(self.resolution)) - int(np.log2(self.fc_size))).
    """

    def ConvLayer(self, in_channel, out_channel):
        """
        helper function to construct (dropout, conv, batchnorm, activations)
        Parameters
        ----------
        in_channel : int
            the number of input channel for Conv3d() layer
        out_channel : int
            the number of out channel for Conv3d() layer
        self.dropout_prob : int
        self.activations : nn actication function
        self.kernel_size : int
            described in Generator Attributes above.
        Returns
        -------
        layer_modules : nn.Sequential(*layer_module)
            a block of layers with (dropout, conv, batchnorm, activations)
        """
        dropout_prob = self.dropout_prob
        activations = self.activations
        kernel_size = self.kernel_size

        layer_module = []
        # dropout
        if dropout_prob > 0:
            layer_module.append(nn.Dropout3d(dropout_prob))

        # convT layer
        if kernel_size == 3:
            layer_module.append(nn.MaxPool3d((2, 2, 2)))
            conv_layer = nn.Conv3d(
                in_channel, out_channel, kernel_size=3, stride=1, padding=1
            )
        elif kernel_size == 4:
            conv_layer = nn.Conv3d(
                in_channel, out_channel, kernel_size=4, stride=2, padding=1
            )
        else:
            conv_layer = nn.Conv3d(
                in_channel, out_channel, kernel_size=5, stride=2, padding=2
            )

        # normalization
        # batch norm
        layer_module.append(conv_layer)
        layer_module.append(nn.BatchNorm3d(out_channel))

        # activation
        layer_module.append(activations)

        return nn.Sequential(*layer_module)

    def __init__(
        self,
        output_size=1,
        resolution=64,
        num_layer_unit=16,
        kernel_size=3,
        fc_size=2,
        dropout_prob=0.0,
        activations=nn.LeakyReLU(0.1, True),
    ):
        super(Discriminator, self).__init__()

        self.output_size = output_size
        self.fc_size = fc_size
        self.resolution = resolution

        self.dropout_prob = dropout_prob
        self.activations = activations
        self.kernel_size = kernel_size

        # number of conv layer in the model
        # if fc_size=2, resolution=32, it should be 4 layers
        self.num_layers = int(np.log2(self.resolution)) - int(np.log2(self.fc_size))
        unit_list_size = self.num_layers + 1

        # check num_layer_unit list
        if type(num_layer_unit) is list:
            if len(num_layer_unit) < unit_list_size:
                message = f"For resolution={resolution}, fc_size={fc_size}, the list of num_layer_unit should have {unit_list_size} elements."
                raise Exception(message)
            if len(num_layer_unit) > unit_list_size:
                num_layer_unit = num_layer_unit[:unit_list_size]
                message = f"For resolution={resolution}, fc_size={fc_size}, the list of num_layer_unit should have {unit_list_size} elements. Trimming num_layer_unit to {num_layer_unit}"
                print(message)
            unit_list = num_layer_unit
        elif type(num_layer_unit) is int:
            unit_list = [num_layer_unit] * unit_list_size
        else:
            raise Exception("num_layer_unit should be int of list of int.")

        # n layer conv/deconv
        dis_module = []
        dis_module.append(
            nn.Conv3d(1, unit_list[0], kernel_size=3, stride=1, padding=1)
        )
        dis_module.append(nn.BatchNorm3d(unit_list[0]))
        dis_module.append(self.activations)
        for i in range(self.num_layers):
            dis_module.append(self.ConvLayer(unit_list[i], unit_list[i + 1]))

        # #final layer
        dis_fc_module = []
        if self.fc_size > 1:
            # 1 layers of fc on latent vector
            num_fc_units = unit_list[-1] * self.fc_size * self.fc_size * self.fc_size
            dis_fc_module.append(nn.Linear(num_fc_units, self.output_size))
        else:
            # add 1 more conv layer
            dis_fc_module.append(
                nn.Conv3d(
                    unit_list[-1], self.output_size, kernel_size=3, stride=1, padding=1
                )
            )

        self.dis_fc_module = nn.Sequential(*dis_fc_module)
        self.dis_module = nn.Sequential(*dis_module)

        # placeholder
        self.dis_fc = nn.Linear(1, 1)

    def forward(self, x, output_all=False):
        """
        default function for nn.Module to run output=model(input)
        defines how the model process data input to output using model components
        Parameters
        ----------
        x : torch.Tensor
            the input data tensor in shape (B, 1, R, R, R)
                B = config.batch_size, R = input_size
        Returns
        -------
        x : torch.Tensor
            the discriminator score/logit of the input data
                in shape (B, output_size)
        """
        x = self.dis_module(x)
        fc = x.view(x.shape[0], -1)
        if self.fc_size > 1:
            # dis_fc_module is fc layer
            x = self.dis_fc_module(fc)
        else:
            # dis_fc_module is conv layer
            x = self.dis_fc_module(x)
            x = x.view(x.shape[0], -1)

        if output_all:
            return x, fc
        return x


###
#       other modules
###
def multiple_components_param(param, num_components):
    if type(param) is int:
        param = [param] * num_components
    elif type(param) is list:
        if len(param) != num_components:
            message = f"parameter should have {num_components} elements."
            raise Exception(message)
    return param
