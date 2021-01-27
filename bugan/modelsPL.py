from bugan.functionsPL import *
import bugan

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from argparse import Namespace, ArgumentParser
from torch.utils.data import DataLoader, TensorDataset

import pkgutil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

DEFAULT_NUM_LAYER_UNIT = [512, 512, 256, 256, 128]
DEFAULT_NUM_LAYER_UNIT_REV = DEFAULT_NUM_LAYER_UNIT.copy()
DEFAULT_NUM_LAYER_UNIT_REV.reverse()


class BaseModel(pl.LightningModule):
    """
    Base model
    The parent of all training models
    This model manages settings that shares among all training models, including
    1) setup model components (optimizer, setup on_epoch / on_batch, logging, ...)
    2) wandb logging (log when epoch end)
    3) GAN hacks (label_noise, instance_noise, dropout_prob, spectral_norm, ...)
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
    noise_magnitude : int
        a placeholder for config.instance_noise (see below)
    instance_noise : None/torch.Tensor
        a placeholder for instance_noise generated per batch (see config.instance_noise_per_batch below)
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
    config.label_flip_prob : float
        create label noise by occassionally flip real/generated labels for Discriminator
        if the value=0.2, this means the label has probability of 0.2 to flip
    config.label_noise : float
        create label noise by adding uniform random noise to the real/generated labels
        if the value=0.2, the real label will be in [0.8, 1], and fake label will be in [0, 0.2]
    config.instance_noise_per_batch : boolean
        whether to create instance noise per batch
        if False, generate new instance noise for every sample, including every generated data and real data
        if True, same noise will be applied to every generated data and real data in the same batch
    config.linear_annealed_instance_noise_epoch : float
        noise linear delay time
        if value is 1000, the magnitude of instance noise will start from config.instance_noise,
            and then linearly delay to 0 in 1000 epochs
    config.instance_noise : float
        the initial instance noise magnitude applied to the real data / generated samples
        similar to config.label_noise, adding uniform random noise to the samples
        if the value=0.2, the value of the array element containing voxels will be in [0.8, 1],
            and value of the array element do not contain voxels will be in [-1, -0.8]
    config.z_size : int
        the latent vector size of the model
        latent vector size determines the size of the generated input, and the size of the compressed latent vector in VAE
    config.activation_leakyReLU_slope : float
        the slope of the leakyReLU activation (see torch leakyReLU)
        leakyReLU activation is used for all layers except the last layer of all models
    config.dropout_prob : float
        the dropout probability of all models (see torch Dropout3D)
        all generator/discriminator use (ConvT/Conv)-BatchNorm-activation-dropout structure
    config.spectral_norm : bool
        whether to apply spectral_norm to the output of the conv layer (see SpectralNorm class below)
        if True, the structure will become SpectralNorm(ConvT/Conv)-BatchNorm-activation-dropout
    config.use_simple_3dgan_struct : bool
        whether to use simple_3dgan_struct for generator/discriminator
        if True, discriminator will conv the input to volume (B,1,1,1,1),
            and generator will start convT from volume (B,Z,1,1,1)
        if False, discriminator will conv to (B,C,k,k,k), k=4, then flatten and connect with fc layer,
            and generator will start connect with fc layer from (B,Z) to (B,C), and then reshape to (B,Ck,k,k,k) to start convT
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        ArgumentParser containing default values for all necessary argument
            The arguments here will be added to config if missing.
            If config already have the arguments, the values won't be replaced.
        Parameters
        ----------
        parent_parser : ArgumentParser
            This will usually be the empty ArgumentParser or the ArgumentParser from the ChildModel.add_model_specific_args()
            Then, the arguments here will be added to this ArgumentParser
        Returns
        -------
        parser : ArgumentParser
            the ArgumentParser with all arguments below.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # basic attributes for training
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--resolution", type=int, default=32)
        # wandb log argument
        parser.add_argument("--log_interval", type=int, default=10)
        parser.add_argument("--log_num_samples", type=int, default=1)

        # label noise
        # real/fake label flip probability
        parser.add_argument("--label_flip_prob", type=float, default=0.0)
        # real/fake label noise magnitude
        parser.add_argument("--label_noise", type=float, default=0.0)

        # instance noise (add noise to real data/generate data)
        # generate instance noise once per batch?
        parser.add_argument("--instance_noise_per_batch", type=bool, default=True)
        # noise linear delay time
        parser.add_argument(
            "--linear_annealed_instance_noise_epoch", type=int, default=1000
        )
        parser.add_argument("--instance_noise", type=float, default=0.0)

        # default Generator/Discriminator parameters
        # latent vector size
        parser.add_argument("--z_size", type=int, default=128)
        # activation default leakyReLU
        parser.add_argument("--activation_leakyReLU_slope", type=float, default=0.0)
        # Dropout probability
        parser.add_argument("--dropout_prob", type=float, default=0.0)
        # spectral_norm
        parser.add_argument("--spectral_norm", type=bool, default=False)
        # use_simple_3dgan_struct
        # reference: https://github.com/xchhuang/simple-pytorch-3dgan
        # basically both G and D will conv the input to volume (1,1,1),
        # which is different from the other one that flatten when (k,k,k), k=4
        parser.add_argument("--use_simple_3dgan_struct", type=bool, default=False)

        return parser

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

        # for instance noise
        self.noise_magnitude = self.config.instance_noise
        self.instance_noise = None

        # whether to log input mesh and reconstructed mesh instead of sample mesh from random z
        self.log_reconstruct = False

    def setup_config_arguments(self, config):
        """
        The add_model_specific_args() function listed all the parameters for the model
            from the config dict given by the user, the necessary parameters may not be there
        This function takes all the default values from the add_model_specific_args() function,
            and then add only the missing parameter values to the user config dict
        (the values in add_model_specific_args() will be overwritten by values in user config dict)
        * this function should be in every child model __init__() right after super().__init__()
        Parameters
        ----------
        config : Namespace
            the config containing only user specified arguments
        Returns
        -------
        config : Namespace
            the config containing all necessary arguments for the Model,
            with default arguments from add_model_specific_args() replaced by user specified arguments
        """
        # add missing default parameters
        parser = self.add_model_specific_args(ArgumentParser())
        args = parser.parse_args([])
        if hasattr(config, "selected_model"):
            default_args_filename = (
                "model_args_default/" + config.selected_model + "_default.json"
            )
            try:
                # replace default argument with stored args file
                data = pkgutil.get_data(bugan, default_args_filename)
                fp = io.BytesIO(data)
                default_args = json.load(fp)
                default_args = Namespace(**default_args)
                args = BaseModel.combine_namespace(args, default_args)
                print(
                    f"file {default_args_filename} found. Using stored arguments as model default."
                )
            except:
                print(
                    f"file {default_args_filename} not found. Using ArgumentParser arguments as model default."
                )

        config = BaseModel.combine_namespace(args, config)
        return config

    def setup_Generator(
        self,
        model_name,
        layer_per_block,
        num_layer_unit,
        optimizer_option,
        learning_rate,
        num_classes=None,
    ):
        """
        function to set Generator for the Model
        this function will be used in ChildModel to set up Generator, its optimizer, and wandbLog data
        Parameters
        ----------
        model_name : string
            model name string is recorded only for logging purpose
            only used when calling wandbLog() to log loss of each module in training_step()
        layer_per_block : int
            the number of ConvT-BatchNorm-activation-dropout layers before upsampling layer
            only for config.use_simple_3dgan_struct=False.
            see also Generator class
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
        """
        function to set Discriminator for the Model
        this function will be used in ChildModel to set up Discriminator, its optimizer, and wandbLog data
        Parameters
        ----------
        model_name : string
            model name string is recorded only for logging purpose
            only used when calling wandbLog() to log loss of each module in training_step()
        layer_per_block : int
            the number of Conv-BatchNorm-activation-dropout layers before pooling layer
            only for config.use_simple_3dgan_struct=False.
            see also Discriminator class
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
        Returns
        -------
        discriminator : nn.Module
        """
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
        num_classes=None,
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
        encoder_layer_per_block : int
            same as the layer_per_block for setup_Discriminator()
            see also Discriminator class
        encoder_num_layer_unit : int/list
            same as the num_layer_unit for setup_Discriminator()
            see also Discriminator class
        decoder_layer_per_block : int
            same as the layer_per_block for setup_Generator()
            see also Generator class
        decoder_num_layer_unit : int/list
            same as the num_layer_unit for setup_Generator()
            see also Generator class
        optimizer_option : string
            the string in ['Adam', 'SGD'], setup the optimizer of the VAE
        learning_rate : float
            the learning_rate of the VAE optimizer
        Returns
        -------
        vae : nn.Module
        """
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

        decoder_input_size = config.z_size
        if num_classes is not None:
            decoder_input_size = decoder_input_size + num_classes

        decoder = Generator(
            layer_per_block=decoder_layer_per_block,
            z_size=decoder_input_size,
            output_size=config.resolution,
            num_layer_unit=decoder_num_layer_unit,
            dropout_prob=config.dropout_prob,
            spectral_norm=config.spectral_norm,
            use_simple_3dgan_struct=config.use_simple_3dgan_struct,
            activations=nn.LeakyReLU(config.activation_leakyReLU_slope, True),
        )

        vae = VAE(encoder=encoder, decoder=decoder, num_classes=num_classes)
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
            loss = (
                lambda gen, data: nn.CrossEntropyLoss(reduction="sum")(gen, data)
                / data.shape[0]
            )
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
            and the instance noise magnitude for every epoch
        Parameters
        ----------
        self.model_list : list of nn.Module
        self.model_ep_loss_list : list of list of numpy array shape [1]
            model_ep_loss_list[i][b] is the loss of the model component with index i in training batch with index b
        self.noise_magnitude : float
            a placeholder for config.instance_noise (see below)
            will be updated here based on the linear delay with parameters
                linear_annealed_instance_noise_epoch and instance_noise
        self.config.linear_annealed_instance_noise_epoch : int
        self.config.instance_noise : float
        """

        # reset ep_loss
        # set model to train
        for idx in range(len(self.model_ep_loss_list)):
            self.model_ep_loss_list[idx] = []
            self.model_list[idx].train()
        # also process other_loss_dict like model_ep_loss_list
        for i in self.other_loss_dict:
            self.other_loss_dict[i] = []

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
        """
        default function for pl.LightningModule to run on the start of every train batch
        here the function just setup the placeholder for instance_noise when config.instance_noise_per_batch=True
        if config.instance_noise_per_batch=False, self.instance_noise will always be None
        Parameters
        ----------
        self.instance_noise : None/torch.Tensor
            a placeholder for instance_noise generated per batch (see self.config.instance_noise_per_batch)
            same noise will be applied to every generated data and real data in the same batch
                when config.instance_noise_per_batch=True
        batch : tuple
        batch_idx : int
        dataloader_idx : int
        """

        # record/reset instance noise per batch
        self.instance_noise = None

    #####
    #   on_train_epoch_end()
    #####
    def on_train_epoch_end(self, epoch_output):
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
            loss_name = self.model_name_list[idx] + " loss"
            log_dict[loss_name] = loss
        # record loss for other_loss_dict like model_ep_loss_list
        for i in self.other_loss_dict:
            loss = np.mean(self.other_loss_dict[i])
            log_dict[i] = loss

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
        add more information into the log_dict to log data/statistics to wandb
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
                    sample_trees = self.generate_tree(c=c, num_trees=log_num_samples)
                    (
                        numpoints,
                        num_cluster,
                        image,
                        voxelmesh,
                        std,
                    ) = calculate_log_media_stat(sample_trees)

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
                    sample_input = self.trainer.datamodule.sample_data(
                        num_samples=log_num_samples
                    )
                    (
                        numpoints,
                        num_cluster,
                        image,
                        voxelmesh,
                        std,
                    ) = calculate_log_media_stat(sample_input)
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
                    sample_trees = self.generate_tree(num_trees=log_num_samples)

                (
                    numpoints,
                    num_cluster,
                    image,
                    voxelmesh,
                    std,
                ) = calculate_log_media_stat(sample_trees)

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
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        dataset_indices : torch.Tensor
            the input data indices for conditional data from the datamodule
                where array is in shape (B,). B = config.batch_size
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
            this function also add noise to the labels,
            so true:[1-label_noise,1], false:[0,label_noise]
            also, flip label with P(config.label_flip_prob)
        Parameters
        ----------
        dataset_batch : torch.Tensor
            data from the dataloader. just for obtaining bach_size and tensor type
        self.config.label_noise : float
            create label noise by adding uniform random noise to the real/generated labels
            if the value=0.2, the real label will be in [0.8, 1], and fake label will be in [0, 0.2]
        self.config.label_flip_prob : float
            create label noise by occassionally flip real/generated labels for Discriminator
            if the value=0.2, this means the label has probability of 0.2 to flip
        Returns
        -------
        real_label : torch.Tensor
            the labels for Discriminator on classifying data from dataset
            the label is scaled to [1-label_noise,1], occassionally flipped to [0,label_noise]
        fake_label : torch.Tensor
            the labels for Discriminator on classifying the generated data from the model
            the label is scaled to [0,label_noise], occassionally flipped to [1-label_noise,1]
        """

        # TODO: refactor to not use dataset_batch, just batch_size
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

    def apply_accuracy_hack(self, dloss, dout_real, dout_fake):
        """
        accuracy hack
        stop update discriminator when prediction accurary > config.accuracy_hack
        stop update by return 0 loss
        * the parameter config.accuracy_hack is not in BaseModel.add_model_specific_args().
            Please add accuracy_hack in the child class.
        Parameters
        ----------
        dloss : torch.Tensor
            the loss of the discriminator from the loss function
        dout_real : torch.Tensor
            the discriminator output of the real data (from dataset)
            the tensor value is assume to be logit (real if value >= 0)
        dout_fake : torch.Tensor
            the discriminator output of the fake data (the generated data from the Generator)
            the tensor value is assume to be logit (fake if value < 0)
        Returns
        -------
        dloss : torch.Tensor
            the new loss of the discriminator after the accuracy hack applied
        """
        config = self.config
        # accuracy hack
        if config.accuracy_hack < 1.0:
            # hack activated, calculate accuracy
            # note that dout are before sigmoid
            real_score = (dout_real >= 0).float()
            fake_score = (dout_fake < 0).float()
            accuracy = torch.cat((real_score, fake_score), 0).mean()
            if accuracy > config.accuracy_hack:
                # TODO: check return loss to stop update is implemented correctly or not
                return dloss - dloss
        return dloss

    @staticmethod
    def merge_latent_and_class_vector(latent_vector, class_vector, num_classes):
        """
        for conditional models
        given latent_vector (B, Z) and class_vector (B),
        reshape class_vector to one-hot (B, num_classes),
        and merge with latent_vector
        Parameters
        ----------
        latent_vector : torch.Tensor
            the latent vector with shape (B, Z)
        class_vector : torch.Tensor
            the class vector with shape (B,), where each value is the class index integer
        num_classes : int
        Returns
        -------
        z : torch.Tensor
            the merged latent vector with shape (B, Z + num_classes)
        """
        z = latent_vector
        c = class_vector
        batch_size = z.shape[0]
        # convert c to one-hot
        c = c.reshape((-1, 1))
        c_onehot = torch.zeros([batch_size, num_classes]).type_as(z)
        c_onehot = c_onehot.scatter(1, c, 1)

        # merge with z to be generator input
        z = torch.cat((z, c_onehot), 1)
        return z

    #
    #
    def add_noise_to_samples(self, data):
        """
        add instance noise to data (real data from dataset / generated data from model)
        create and add instance noise which has the same shape as the data
        Parameters
        ----------
        data : torch.Tensor
            the data to add instance noise
        self.config.instance_noise_per_batch : boolean
            whether to create instance noise per batch
            if False, generate new instance noise for every sample, including every generated data and real data
            if True, same noise will be applied to every generated data and real data in the same batch
        self.noise_magnitude : float
            a placeholder for config.instance_noise, will be updated on_train_epoch_start()
            create instance noise by adding uniform random noise to the value of the mesh array
            if the value=0.2, the value of index contain voxel will be in [0.8, 1],
                and value of index contain no voxel will be in [-1, -0.8]
        self.instance_noise : None/torch.Tensor
            a placeholder for instance_noise generated per batch
            if self.config.instance_noise_per_batch=True, the generated noise will be recorded in self.instance_noise
                and applied to every data in the same batch (will reset to None when next batch start).
            if self.config.instance_noise_per_batch=False, self.instance_noise should always be None.
        Returns
        -------
        data : torch.Tensor
            data + instance noise
        """
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

    # TODO: change generate tree to generate samples to avoid confusion
    def generate_tree(self, generator, c=None, num_classes=None, num_trees=1):
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

# TODO: doc about KL loss coef and voxel diff coef
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
        dictionary of training parameters
    config.vae_decoder_layer : int
        the decoder_layer_per_block for BaseModel setup_VAE(). Only for config.use_simple_3dgan_struct=False.
        see also Generator class, BaseModel setup_VAE()
    config.vae_encoder_layer : int
        the encoder_layer_per_block for BaseModel setup_VAE(). Only for config.use_simple_3dgan_struct=False.
        see also Discriminator class, BaseModel setup_VAE()
    config.vae_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the VAE
        Using 'Adam' may cause VAE KL loss go to inf.
    config.rec_loss : string
        rec_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assume input to be logit (before sigmoid/tanh)
    config.vae_lr : float
        the learning_rate of the VAE
    config.kl_coef : float
        the coefficient of the KL loss in the final VAE loss
    config.voxel_diff_coef : float
        the coefficient of the voxel difference loss in the final VAE loss
        added absolute voxel difference between data and reconstructed as a part of loss
    config.decoder_num_layer_unit : int/list
        the decoder_num_layer_unit for BaseModel setup_VAE()
        see also Generator class, BaseModel setup_VAE()
    config.encoder_num_layer_unit : int/list
        the encoder_num_layer_unit for BaseModel setup_VAE()
        see also Discriminator class, BaseModel setup_VAE()
    self.vae : nn.Module
        the model component from setup_VAE()
    self.criterion_reconstruct : nn Loss function
        the loss function based on config.rec_loss
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        ArgumentParser containing default values for all necessary argument
            The arguments here will be added to config if missing.
            If config already have the arguments, the values won't be replaced.
        Parameters
        ----------
        parent_parser : ArgumentParser
            This will usually be the empty ArgumentParser or the ArgumentParser from the ChildModel.add_model_specific_args()
            Then, the arguments here will be added to this ArgumentParser
        Returns
        -------
        parser : ArgumentParser
            the ArgumentParser with all arguments below.
        """
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
        parser.add_argument("--vae_lr", type=float, default=1e-5)
        # KL loss coefficient
        parser.add_argument("--kl_coef", type=float, default=1)
        # number of unit per layer
        decoder_num_layer_unit = DEFAULT_NUM_LAYER_UNIT
        encoder_num_layer_unit = DEFAULT_NUM_LAYER_UNIT_REV
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

        # log input mesh and reconstructed mesh
        self.log_reconstruct = True

        # loss function is modified to handle pos_weight
        if config.rec_loss == "BCELoss":
            self.criterion_reconstruct = (
                lambda gen, data: nn.BCEWithLogitsLoss(
                    reduction="sum", pos_weight=self.calculate_pos_weight(data)
                )(gen, data)
                / data.shape[0]
            )
        else:
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
        assume BCELoss: data in [0,1], and MSELoss: data in [-1,1]
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
            the shape of pos_weight for BCELoss is a number (shape [1])
            the shape of pos_weight for MSELoss is (B, 1, res, res, res)
            (negative weigth = 1, positive weight = num_zeros/num_ones)
        """
        if self.config.rec_loss == "BCELoss":
            # assume data in [0,1]
            num_ones = torch.sum(data)
            num_zeros = torch.sum(1 - data)
            pos_weight = num_zeros / num_ones
        else:
            # assume data in [-1,1], scale back to [0,1]
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
            the loss function based on config.rec_loss to calculate the loss of VAE
        Returns
        -------
        vae_loss : torch.Tensor of shape [1]
            the loss of the VAE.
        """
        config = self.config
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        batch_size = dataset_batch.shape[0]

        reconstructed_data, z, mu, logVar = self.vae(dataset_batch, output_all=True)
        # add instance noise
        reconstructed_data = self.add_noise_to_samples(reconstructed_data)

        # for BCELoss, the "target" should be in [0,1]
        if config.rec_loss == "BCELoss":
            dataset_batch = (dataset_batch + 1) / 2
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
    config.vae_decoder_layer : int
        the decoder_layer_per_block for BaseModel setup_VAE(). Only for config.use_simple_3dgan_struct=False.
        see also Generator class, BaseModel setup_VAE()
    config.vae_encoder_layer : int
        the encoder_layer_per_block for BaseModel setup_VAE(). Only for config.use_simple_3dgan_struct=False.
        see also Discriminator class, BaseModel setup_VAE()
    config.d_layer : int
        the layer_per_block for BaseModel setup_Discriminator(). Only for config.use_simple_3dgan_struct=False.
        see also Discriminator class, BaseModel setup_Discriminator()
    config.vae_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the VAE
        Using 'Adam' may cause VAE KL loss go to inf.
    config.dis_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the Discriminator
    config.label_loss : string
        rec_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assuming input to be logit (before sigmoid/tanh)
        this is the prediction loss for discriminator on classifying reconstructed data and real data
    config.rec_loss : string
        rec_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assuming input to be logit (before sigmoid/tanh)
    config.accuracy_hack : float
        the accuracy threshold of the Discriminator to stop update
        if the accuracy of Discriminator on classifying real data from generated data
            is larger than the threshold, the Discriminator stop updating parameters
            on that batch
    config.vae_lr : float
        the learning_rate of the VAE
    config.d_lr : float
        the learning_rate of the Discriminator
    config.kl_coef : float
        the coefficient of the KL loss in the final VAE loss
    config.d_rec_coef : float
        the coefficient of the Discriminator loss compared to the reconstruction loss
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
        the loss function based on config.rec_loss
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        ArgumentParser containing default values for all necessary argument
            The arguments here will be added to config if missing.
            If config already have the arguments, the values won't be replaced.
        Parameters
        ----------
        parent_parser : ArgumentParser
            This will usually be the empty ArgumentParser or the ArgumentParser from the ChildModel.add_model_specific_args()
            Then, the arguments here will be added to this ArgumentParser
        Returns
        -------
        parser : ArgumentParser
            the ArgumentParser with all arguments below.
        """
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
        parser.add_argument("--vae_lr", type=float, default=1e-5)
        parser.add_argument("--d_lr", type=float, default=1e-5)
        # KL loss coefficient
        parser.add_argument("--kl_coef", type=float, default=1)
        # Discriminator loss coefficient (compared to rec_loss)
        parser.add_argument("--d_rec_coef", type=float, default=1)

        # number of unit per layer
        decoder_num_layer_unit = DEFAULT_NUM_LAYER_UNIT
        encoder_num_layer_unit = DEFAULT_NUM_LAYER_UNIT_REV
        dis_num_layer_unit = DEFAULT_NUM_LAYER_UNIT_REV

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
        # rec loss function is modified to handle pos_weight
        if config.rec_loss == "BCELoss":
            self.criterion_reconstruct = (
                lambda gen, data: nn.BCEWithLogitsLoss(
                    reduction="sum", pos_weight=self.calculate_pos_weight(data)
                )(gen, data)
                / data.shape[0]
            )
        else:
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
        assume BCELoss: data in [0,1], and MSELoss: data in [-1,1]
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
            the shape of pos_weight for BCELoss is a number (shape [1])
            the shape of pos_weight for MSELoss is (B, 1, res, res, res)
            (negative weigth = 1, positive weight = num_zeros/num_ones)
        """
        if self.config.rec_loss == "BCELoss":
            # assume data in [0,1]
            num_ones = torch.sum(data)
            num_zeros = torch.sum(1 - data)
            pos_weight = num_zeros / num_ones
        else:
            # assume data in [-1,1], scale back to [0,1]
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
            the loss function based on config.rec_loss to calculate the loss of VAE
        Returns
        -------
        vae_loss : torch.Tensor of shape [1]
            the loss of the VAE.
        dloss : torch.Tensor of shape [1]
            the loss of the discriminator.
        """

        config = self.config
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        if optimizer_idx == 0:
            ############
            #   VAE
            ############

            reconstructed_data_logit, z, mu, logVar = self.vae(
                dataset_batch, output_all=True
            )

            # for BCELoss, the "target" should be in [0,1]
            if config.rec_loss == "BCELoss":
                vae_rec_loss = self.criterion_reconstruct(
                    reconstructed_data_logit, (dataset_batch + 1) / 2
                )
            else:
                vae_rec_loss = self.criterion_reconstruct(
                    reconstructed_data_logit, dataset_batch
                )
            self.record_loss(vae_rec_loss.detach().cpu().numpy(), loss_name="rec_loss")

            # add KL loss
            KL = self.vae.calculate_log_prob_loss(z, mu, logVar) * config.kl_coef
            self.record_loss(KL.detach().cpu().numpy(), loss_name="KL_loss")
            vae_rec_loss += KL

            # output of the vae should fool discriminator
            vae_out_d1 = self.discriminator(F.tanh(reconstructed_data_logit))
            vae_d_loss1 = self.criterion_label(vae_out_d1, real_label)
            ##### generate fake trees
            latent_size = self.vae.decoder_z_size
            # latent noise vector
            z = torch.randn(batch_size, latent_size).float().type_as(dataset_batch)
            tree_fake = F.tanh(self.vae.generate_sample(z))
            # output of the vae should fool discriminator
            vae_out_d2 = self.discriminator(tree_fake)
            vae_d_loss2 = self.criterion_label(vae_out_d2, real_label)

            vae_d_loss = (vae_d_loss1 + vae_d_loss2) / 2

            vae_d_loss = vae_d_loss * config.d_rec_coef
            self.record_loss(vae_d_loss.detach().cpu().numpy(), loss_name="vae_d_loss")
            vae_loss = (vae_rec_loss + vae_d_loss) / 2

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

            # fake data (data from generator)
            # detach so no update to generator
            dout_fake = self.discriminator(tree_fake.clone().detach())
            dloss_fake = self.criterion_label(dout_fake, fake_label)
            # real data (data from dataloader)
            dout_real = self.discriminator(dataset_batch)
            dloss_real = self.criterion_label(dout_real, real_label)

            dloss = (dloss_fake + dloss_real) / 2  # scale the loss to one

            # accuracy hack
            dloss = self.apply_accuracy_hack(dloss, dout_real, dout_fake)
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
    config.g_layer : int
        the layer_per_block for BaseModel setup_Generator(). Only for config.use_simple_3dgan_struct=False.
        see also Generator class, BaseModel setup_VAE())
    config.d_layer : int
        the layer_per_block for BaseModel setup_Discriminator(). Only for config.use_simple_3dgan_struct=False.
        see also Discriminator class, BaseModel setup_Discriminator()
    config.gen_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the Generator
    config.dis_opt : string
        the string in ['Adam', 'SGD'], setup the optimizer of the Discriminator
    config.label_loss : string
        rec_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assuming input to be logit (before sigmoid/tanh)
        this is the prediction loss for discriminator and generator
    config.accuracy_hack : float
        the accuracy threshold of the Discriminator to stop update
        if the accuracy of Discriminator on classifying real data from generated data
            is larger than the threshold, the Discriminator stop updating parameters
            on that batch
    config.g_lr : float
        the learning_rate of the Generator
    config.d_lr : float
        the learning_rate of the Discriminator
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
    def add_model_specific_args(parent_parser):
        """
        ArgumentParser containing default values for all necessary argument
            The arguments here will be added to config if missing.
            If config already have the arguments, the values won't be replaced.
        Parameters
        ----------
        parent_parser : ArgumentParser
            This will usually be the empty ArgumentParser or the ArgumentParser from the ChildModel.add_model_specific_args()
            Then, the arguments here will be added to this ArgumentParser
        Returns
        -------
        parser : ArgumentParser
            the ArgumentParser with all arguments below.
        """
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
        gen_num_layer_unit = DEFAULT_NUM_LAYER_UNIT
        dis_num_layer_unit = DEFAULT_NUM_LAYER_UNIT_REV

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

            # accuracy hack
            dloss = self.apply_accuracy_hack(dloss, dout_real, dout_fake)
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
    def add_model_specific_args(parent_parser):
        """
        ArgumentParser containing default values for all necessary argument
            The arguments here will be added to config if missing.
            If config already have the arguments, the values won't be replaced.
        Parameters
        ----------
        parent_parser : ArgumentParser
            This will usually be the empty ArgumentParser or the ArgumentParser from the ChildModel.add_model_specific_args()
            Then, the arguments here will be added to this ArgumentParser
        Returns
        -------
        parser : ArgumentParser
            the ArgumentParser with all arguments below.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # important argument
        parser.add_argument("--clip_value", type=float, default=0.01)

        return GAN.add_model_specific_args(parser)

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
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)
        batch_size = dataset_batch.shape[0]

        # label no used in WGAN
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
    def add_model_specific_args(parent_parser):
        """
        ArgumentParser containing default values for all necessary argument
            The arguments here will be added to config if missing.
            If config already have the arguments, the values won't be replaced.
        Parameters
        ----------
        parent_parser : ArgumentParser
            This will usually be the empty ArgumentParser or the ArgumentParser from the ChildModel.add_model_specific_args()
            Then, the arguments here will be added to this ArgumentParser
        Returns
        -------
        parser : ArgumentParser
            the ArgumentParser with all arguments below.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # important argument
        parser.add_argument("--gp_epsilon", type=float, default=2.0)

        return GAN.add_model_specific_args(parser)

    def gradient_penalty(self, real_tree, generated_tree):
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
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)
        batch_size = dataset_batch.shape[0]

        # label no used in Wloss
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
            return gloss

        if optimizer_idx == 1:

            ############
            #   discriminator
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
        rec_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
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
    def add_model_specific_args(parent_parser):
        """
        ArgumentParser containing default values for all necessary argument
            The arguments here will be added to config if missing.
            If config already have the arguments, the values won't be replaced.
        Parameters
        ----------
        parent_parser : ArgumentParser
            This will usually be the empty ArgumentParser or the ArgumentParser from the ChildModel.add_model_specific_args()
            Then, the arguments here will be added to this ArgumentParser
        Returns
        -------
        parser : ArgumentParser
            the ArgumentParser with all arguments below.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # log argument
        parser.add_argument("--num_classes", type=int, default=10)
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--class_loss", type=str, default="CrossEntropyLoss")

        return GAN.add_model_specific_args(parser)

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
        z = self.merge_latent_and_class_vector(x, c, self.config.num_classes)

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
            # class vector
            c_fake = (
                torch.randint(0, config.num_classes, (batch_size,))
                .type_as(dataset_batch)
                .to(torch.int64)
            )

            # combine z and c_fake
            z = self.merge_latent_and_class_vector(z, c_fake, self.config.num_classes)

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
            z = self.merge_latent_and_class_vector(z, c, self.config.num_classes)

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

            # accuracy hack
            dloss = self.apply_accuracy_hack(dloss, dout_real, dout_fake)
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
            # z = self.merge_latent_and_class_vector(z, c_fake, self.config.num_classes)

            # # detach so no update to generator
            # tree_fake = F.tanh(self.generator(z)).clone().detach()
            # # add noise to data
            # tree_fake = self.add_noise_to_samples(tree_fake)

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
            generator, c=c, num_classes=config.num_classes, num_trees=num_trees
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
        rec_loss in ['BCELoss', 'MSELoss', 'CrossEntropyLoss']
        the returned loss assuming input to be logit (before sigmoid/tanh)
        this is the classification loss for classifier
    config.num_classes : int
        the number of classes in the dataset/datamodule
        if the datamodule is DataModule_process class, the config.num_classes there should be the same
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
        the loss function based on config.rec_loss
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        ArgumentParser containing default values for all necessary argument
            The arguments here will be added to config if missing.
            If config already have the arguments, the values won't be replaced.
        Parameters
        ----------
        parent_parser : ArgumentParser
            This will usually be the empty ArgumentParser or the ArgumentParser from the ChildModel.add_model_specific_args()
            Then, the arguments here will be added to this ArgumentParser
        Returns
        -------
        parser : ArgumentParser
            the ArgumentParser with all arguments below.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # log argument
        parser.add_argument("--num_classes", type=int, default=10)
        # loss function in {'BCELoss', 'MSELoss', 'CrossEntropyLoss'}
        parser.add_argument("--class_loss", type=str, default="CrossEntropyLoss")
        # Classifier loss coefficient (compared to rec_loss)
        parser.add_argument("--c_rec_coef", type=float, default=1)

        return VAEGAN.add_model_specific_args(parser)

    def __init__(self, config):
        super(VAEGAN, self).__init__(config)
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
        # rec loss function is modified to handle pos_weight
        if config.rec_loss == "BCELoss":
            self.criterion_reconstruct = (
                lambda gen, data: nn.BCEWithLogitsLoss(
                    reduction="sum", pos_weight=self.calculate_pos_weight(data)
                )(gen, data)
                / data.shape[0]
            )
        else:
            self.criterion_reconstruct = (
                lambda gen, data: torch.sum(
                    nn.MSELoss(reduction="none")(F.tanh(gen), data)
                    * self.calculate_pos_weight(data)
                )
                / data.shape[0]
            )

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
            the loss function based on config.rec_loss to calculate the loss of VAE
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
        # add noise to data
        dataset_batch = self.add_noise_to_samples(dataset_batch)

        real_label, fake_label = self.create_real_fake_label(dataset_batch)
        batch_size = dataset_batch.shape[0]

        if optimizer_idx == 0:
            ############
            #   generator
            ############

            reconstructed_data_logit, z, mu, logVar = self.vae(
                dataset_batch, dataset_indices, output_all=True
            )

            # for BCELoss, the "target" should be in [0,1]
            if config.rec_loss == "BCELoss":
                vae_rec_loss = self.criterion_reconstruct(
                    reconstructed_data_logit, (dataset_batch + 1) / 2
                )
            else:
                vae_rec_loss = self.criterion_reconstruct(
                    reconstructed_data_logit, dataset_batch
                )
            self.record_loss(vae_rec_loss.detach().cpu().numpy(), loss_name="rec_loss")

            # add KL loss
            KL = self.vae.calculate_log_prob_loss(z, mu, logVar) * config.kl_coef
            self.record_loss(KL.detach().cpu().numpy(), loss_name="KL loss")
            vae_rec_loss += KL

            # output of the vae should fool discriminator
            vae_out_d1 = self.discriminator(F.tanh(reconstructed_data_logit))
            vae_d_loss1 = self.criterion_label(vae_out_d1, real_label)
            ##### generate fake trees
            # latent noise vector
            z = torch.randn(batch_size, config.z_size).float().type_as(dataset_batch)
            z = self.merge_latent_and_class_vector(
                z, dataset_indices, self.config.num_classes
            )
            tree_fake = F.tanh(self.vae.generate_sample(z))
            # output of the vae should fool discriminator
            vae_out_d2 = self.discriminator(tree_fake)
            vae_d_loss2 = self.criterion_label(vae_out_d2, real_label)
            vae_d_loss = (vae_d_loss1 + vae_d_loss2) / 2

            # tree_fake on Cla
            vae_out_c = self.classifier(F.tanh(reconstructed_data_logit))
            vae_c_loss = self.criterion_class(vae_out_c, dataset_indices)

            vae_d_loss = vae_d_loss * config.d_rec_coef
            self.record_loss(vae_d_loss.detach().cpu().numpy(), loss_name="vae_d_loss")
            vae_c_loss = vae_c_loss * config.c_rec_coef
            self.record_loss(vae_c_loss.detach().cpu().numpy(), loss_name="vae_c_loss")
            vae_loss = (vae_rec_loss + vae_d_loss + vae_c_loss) / 3

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
            z = self.merge_latent_and_class_vector(z, c, self.config.num_classes)

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

            # accuracy hack
            dloss = self.apply_accuracy_hack(dloss, dout_real, dout_fake)
            return dloss

        if optimizer_idx == 2:

            ############
            #   classifier
            ############

            # reconstructed_data = self.vae(dataset_batch, dataset_indices)
            # # add noise to data
            # reconstructed_data = self.add_noise_to_samples(F.tanh(reconstructed_data).clone().detach())

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
            generator, c=c, num_classes=config.num_classes, num_trees=num_trees
        )


#####
#   building blocks for models
#####


class Generator(nn.Module):
    """
    The generator class
    This class serves as the generator and VAE decoder of models above
    Attributes
    ----------
    layer_per_block : int
        the number of ConvT-BatchNorm-activation-dropout layers before upsampling layer
            only for config.use_simple_3dgan_struct=False.
    z_size : int
        the latent vector size of the model
        latent vector size determines the size of the generated input, and the size of the compressed latent vector in VAE
    output_size : int
        the output size of the generator
        given latent vector (B,Z), the output will be (B,1,R,R,R)
            Z = z_size, R = output_size, B = batch_size
    num_layer_unit : int/list
        the number of unit in the ConvT layer
            if int, every ConvT will have the same specified number of unit.
            if list, every ConvT layer between the upsampling layers will have the specified number of unit.
    dropout_prob : float
        the dropout probability of the generator models (see torch Dropout3D)
        the generator use ConvT-BatchNorm-activation-dropout structure
    spectral_norm : boolean
        whether to apply spectral_norm to the output of the conv layer (see SpectralNorm class below)
        if True, the structure will become SpectralNorm(ConvT/Conv)-BatchNorm-activation-dropout
    use_simple_3dgan_struct : boolean
        whether to use simple_3dgan_struct for generator/discriminator
        if True, discriminator will conv the input to volume (B,1,1,1,1),
            and generator will start convT from volume (B,Z,1,1,1)
        if False, discriminator will conv to (B,C,k,k,k), k=4, then flatten and connect with fc layer,
            and generator will start connect with fc layer from (B,Z) to (B,C), and then reshape to (B,Ck,k,k,k) to start convT
    activations : nn actication function
        the actication function used for all layers except the last layer of all models
    """

    def __init__(
        self,
        layer_per_block=1,
        z_size=128,
        output_size=64,
        num_layer_unit=32,
        dropout_prob=0.0,
        spectral_norm=False,
        use_simple_3dgan_struct=False,
        activations=nn.LeakyReLU(0.0, True),
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
            B = config.batch_size, R = output_size
        """
        if not self.use_simple_3dgan_struct:
            x = self.gen_fc(x)
        x = x.view(
            x.shape[0], self.fc_channel, self.fc_size, self.fc_size, self.fc_size
        )
        x = self.gen(x)
        return x


class Discriminator(nn.Module):
    """
    The Discriminator class
    This class serves as the discriminator, the classifier, and VAE encoder of models above
    Attributes
    ----------
    layer_per_block : int
        the number of ConvT-BatchNorm-activation-dropout layers before upsampling layer
            only for config.use_simple_3dgan_struct=False.
    output_size : int
        the output size of the discriminator
        the input will be in the shape (B,S)
            S = output_size, B = batch_size
    input_size : int
        the input size of the discriminator
        the input will be in the shape (B,1,R,R,R)
            R = input_size, B = batch_size
    num_layer_unit : int/list
        the number of unit in the ConvT layer
            if int, every ConvT will have the same specified number of unit.
            if list, every ConvT layer between the upsampling layers will have the specified number of unit.
    dropout_prob : float
        the dropout probability of the generator models (see torch Dropout3D)
        the generator use ConvT-BatchNorm-activation-dropout structure
    spectral_norm : boolean
        whether to apply spectral_norm to the output of the conv layer (see SpectralNorm class below)
        if True, the structure will become SpectralNorm(ConvT/Conv)-BatchNorm-activation-dropout
    use_simple_3dgan_struct : boolean
        whether to use simple_3dgan_struct for generator/discriminator
        if True, discriminator will conv the input to volume (B,1,1,1,1),
            and generator will start convT from volume (B,Z,1,1,1)
        if False, discriminator will conv to (B,C,k,k,k), k=4, then flatten and connect with fc layer,
            and generator will start connect with fc layer from (B,Z) to (B,C), and then reshape to (B,Ck,k,k,k) to start convT
    activations : nn actication function
        the actication function used for all layers except the last layer of all models
    """

    def __init__(
        self,
        layer_per_block=1,
        output_size=1,
        input_size=64,
        num_layer_unit=16,
        dropout_prob=0.0,
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
                num_layer_unit = num_layer_unit[-self.num_blocks :]
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
        x = self.dis(x)
        x = x.view(x.shape[0], -1)
        if not self.use_simple_3dgan_struct:
            x = self.dis_fc(x)
        return x


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

    def __init__(self, encoder, decoder, num_classes=None, dropout_prob=0.0):
        super(VAE, self).__init__()
        assert encoder.input_size == decoder.output_size
        self.sample_size = decoder.output_size
        self.encoder_z_size = encoder.output_size
        self.decoder_z_size = decoder.z_size
        self.num_classes = num_classes
        if num_classes is not None:
            assert (self.decoder_z_size - self.encoder_z_size) == self.num_classes
        # VAE
        self.vae_encoder = encoder
        self.encoder_output_dropout = nn.Dropout(dropout_prob)

        self.encoder_mean = nn.Linear(self.encoder_z_size, self.encoder_z_size)
        self.encoder_logvar = nn.Linear(self.encoder_z_size, self.encoder_z_size)
        self.encoder_mean_dropout = nn.Dropout(dropout_prob)
        self.encoder_logvar_dropout = nn.Dropout(dropout_prob)

        self.vae_decoder = decoder

    def noise_reparameterize(self, mean, logvar):
        """
        noise reparameterization of the VAE
        reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        reference: https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/cvae.ipynb
        Parameters
        ----------
        mean : torch.Tensor of shape (B, Z)
        logvar : torch.Tensor of shape (B, Z)
        """
        # # reference: https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner/blob/master/CVAE-GAN/CVAE-GAN.py
        # # reference: https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/cvae.ipynb
        # eps = torch.randn(mean.shape).type_as(mean)
        # z = mean + eps * torch.exp(logvar / 2.0)

        # reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z

    def calculate_log_prob_loss(self, z, mu, logVar):
        """
        calculate log_prob loss (KL/ELBO??) for VAE based on the mean and logvar used for noise_reparameterize()
        reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        reference: https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/cvae.ipynb
        See VAE class
        Parameters
        ----------
        mu : torch.Tensor
        logVar : torch.Tensor
        """
        # # reference: https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner/blob/master/CVAE-GAN/CVAE-GAN.py
        # loss = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1.0 - logVar)

        # reference: https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        std = torch.exp(logVar / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        kl = log_qz - log_pz
        loss = kl.mean()

        # # reference: https://github.com/tensorflow/docs-l10n/blob/master/site/zh-cn/tutorials/generative/cvae.ipynb
        # log2pi = torch.log(2 * torch.tensor(np.pi))
        # logpz = torch.sum(0.5 * (z ** 2 + log2pi), axis=1)
        # logqz_x = torch.sum(
        #     0.5 * ((z - mu) ** 2.0 * torch.exp(-logVar) + logVar + log2pi), axis=1
        # )
        # loss = torch.mean(logpz - logqz_x)
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
        f = self.encoder_output_dropout(f)

        x_mean = self.encoder_mean(f)
        x_logvar = self.encoder_logvar(f)

        x_mean = self.encoder_mean_dropout(x_mean)
        x_logvar = self.encoder_logvar_dropout(x_logvar)

        z = self.noise_reparameterize(x_mean, x_logvar)

        # handle class vector
        if c is not None:
            x = BaseModel.merge_latent_and_class_vector(z, c, self.num_classes)
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
            z = BaseModel.merge_latent_and_class_vector(z, c, self.num_classes)

        x = self.vae_decoder(z)
        return x


###
#       other modules
###


class SpectralNorm(nn.Module):
    """
    the module to perform SpectralNorm to tensor
    Copied from: https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    """

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
