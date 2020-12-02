#!/usr/bin/env python
import argparse
import logging
import os
import pprint

import configargparse
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from bugan.functionsPL import (
    DataModule_process,
    DataModule_custom,
    load_checkpoint_from_cloud,
)
from bugan.modelsPL import VAEGAN

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

parser = configargparse.ArgumentParser(
    auto_env_var_prefix="GAN_",
    add_config_file_help=False,
    add_env_var_help=False,
)

parser.add(
    "-c",
    "--config",
    required=False,
    is_config_file=True,
    help="Path to configuration file.",
)
parser.add("--data-path", "-d", required=True, help="Dataset directory.")
parser.add("--id", required=False, help="Wandb run ID to resume.")
parser.add("--run-path", required=False, help="Wandb run path.")
parser.add("--epochs", default=2000, type=int, help="Number of epochs to train.")
parser.add("--gpus", default=None, type=int, help="Number of GPUs to train with.")

args = parser.parse_args()


if args.id is None:
    resume = False
    args.id = wandb.util.generate_id()
else:
    resume = True

config_dict = dict(
    aug_rotation_type="random rotation",
    batch_size=8,
    resolution=32,
    z_size=128,
    gen_num_layer_unit=[256, 1024, 512, 128],
    dis_num_layer_unit=[32, 64, 128, 128],
    leakyReLU=False,  # leakyReLU implementation still not in modelPL,
    balance_voxel_in_space=False,
    # epochs=2000,
    vae_lr=0.0025,
    vae_encoder_layer=1,
    vae_decoder_layer=2,
    d_lr=0.00005,
    d_layer=1,
    vae_recon_loss_factor=1,
    seed=1234,
    log_interval=5,
    log_num_samples=3,
    data_augmentation=True,
    vae_opt="Adam",
    dis_opt="Adam",
)

config_dict.update(args.__dict__)

run = wandb.init(
    project="tree-gan",
    id=args.id,
    entity="bugan",
    resume=True,
    dir=args.run_path,
    config=config_dict,
)

print(f"Run id: {wandb.run.id}")
print(f"Run name: {wandb.run.name}")
wandb.watch_called = False

config = wandb.config
pprint.pprint(config)

np.random.seed(config["seed"])

dataModule = DataModule_process(config, run, args.data_path)
config.num_data = dataModule.size

torch.manual_seed(config.seed)
torch.autograd.set_detect_anomaly(True)

wandb_logger = WandbLogger(experiment=run, log_model=True)

checkpoint_path = os.path.join(wandb.run.dir, "checkpoint.ckpt")

if resume:
    # get file from the wandb cloud
    load_checkpoint_from_cloud(checkpoint_path="checkpoint.ckpt")
    # restore training state completely
    resume_from_checkpoint = checkpoint_path
else:
    resume_from_checkpoint = None

trainer = pl.Trainer(
    gpus=args.gpus,
    max_epochs=config.epochs,
    logger=wandb_logger,
    checkpoint_callback=None,
    resume_from_checkpoint=resume_from_checkpoint,
    default_root_dir=wandb.run.dir,
    distributed_backend="ddp",
)

# model
frozen_config = argparse.Namespace(**config)
vaegan = VAEGAN(frozen_config)
wandb_logger.watch(vaegan)

trainer.fit(vaegan, dataModule)
