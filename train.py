#!/usr/bin/env python
import io
import logging
import os
import pprint

import configargparse
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

from bugan.functionsPL import DataModule_custom, load_checkpoint_from_cloud
from bugan.modelsPL import VAE, VAEGAN, Discriminator, Generator

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

parser = configargparse.ArgumentParser(
    auto_env_var_prefix="GAN_", add_config_file_help=False, add_env_var_help=False,
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

args = parser.parse_args()

pprint.pprint(args)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.id is None:
    resume = False
    args.id = wandb.util.generate_id()
else:
    resume = True

config_dict = dict(
    batch_size=8,
    array_size=32,
    z_size=128,
    gen_num_layer_unit=[256, 1024, 512, 128],
    dis_num_layer_unit=[32, 64, 128, 128],
    leakyReLU=False,  # leakyReLU implementation still not in modelPL,
    balance_voxel_in_space=False,
    epochs=2000,
    vae_lr=0.0025,
    vae_encoder_layer=1,
    vae_decoder_layer=2,
    d_lr=0.00005,
    d_layer=1,
    vae_recon_loss_factor=1,
    seed=1234,
    log_image_interval=5,
    log_mesh_interval=50,
    data_augmentation=True,
    num_augment_data=4,
    vae_opt="Adam",
    dis_opt="Adam",
)
run = wandb.init(
    project="tree-gan",
    id=args.id,
    entity="bugan",
    resume=True,
    dir=args.run_path,
    config=config_dict,
)

print("run id: " + str(wandb.run.id))
print("run name: " + str(wandb.run.name))
wandb.watch_called = False

config = wandb.config

np.random.seed(config["seed"])

process_data = False if args.data_path.endswith(".npy") else True
dataModule = DataModule_custom(config, run, args.data_path, process_data=process_data)
config.num_data = dataModule.size

torch.manual_seed(config.seed)
torch.autograd.set_detect_anomaly(True)

wandb_logger = WandbLogger(experiment=run, log_model=True)

checkpoint_path = os.path.join(wandb.run.dir, "checkpoint.ckpt")

if resume:
    # get file from the wandb cloud
    load_checkpoint_from_cloud(checkpoint_path="checkpoint.ckpt")
    # restore training state completely
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        checkpoint_callback=None,
        resume_from_checkpoint=checkpoint_path,
    )
else:
    trainer = pl.Trainer(
        max_epochs=config.epochs, logger=wandb_logger, checkpoint_callback=None
    )

# model
vaegan = VAEGAN(config, trainer, save_model_path=checkpoint_path).to(device)
wandb_logger.watch(vaegan)

trainer.fit(vaegan, dataModule)
