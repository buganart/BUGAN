import io
import os
from io import BytesIO
import sys
import zipfile
import trimesh
import numpy as np
from argparse import Namespace, ArgumentParser
import wandb
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from bugan.functionsPL import load_checkpoint_from_cloud, SaveWandbCallback
from bugan.datamodulePL import DataModule_process
from bugan.modelsPL import VAEGAN, VAE_train, GAN, GAN_Wloss, GAN_Wloss_GP, CGAN


def init_wandb_run(config, run_dir="./"):
    resume_id = config.resume_id
    project_name = config.project_name
    selected_model = config.selected_model
    entity = "bugan"
    run_dir = Path(run_dir).absolute()

    if resume_id:
        run_id = resume_id
        # all config will be replaced by the stored one in wandb
        api = wandb.Api()
        previous_run = api.run(f"{entity}/{project_name}/{resume_id}")
        config = Namespace(**previous_run.config)
        # selected_model may be not in config
        if hasattr(config, "selected_model"):
            selected_model = config.selected_model
    else:
        run_id = wandb.util.generate_id()

    run = wandb.init(
        project=project_name,
        id=run_id,
        entity=entity,
        resume=True,
        dir=run_dir,
        group=selected_model,
    )

    print("run id: " + str(wandb.run.id))
    print("run name: " + str(wandb.run.name))
    wandb.watch_called = False
    run.tags.append(selected_model)
    return run, config


def setup_datamodule(config, run):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset_path = Path(config.data_location)
    if config.data_location.endswith(".zip"):
        config.dataset = dataset_path.stem
    else:
        config.dataset = "dataset_array_custom"

    dataModule = DataModule_process(config, run, dataset_path)

    print("dataset name: ", config.dataset)
    print("dataset path: ", dataset_path)
    return dataModule, config


def setup_model(config, run):
    checkpoint_path = str(Path(run.dir).absolute() / "checkpoint.ckpt")
    selected_model = config.selected_model
    # model
    if selected_model == "VAEGAN":
        MODEL_CLASS = VAEGAN
    elif selected_model == "GAN":
        MODEL_CLASS = GAN
    elif selected_model == "VAE":
        MODEL_CLASS = VAE_train
    elif selected_model == "WGAN":
        MODEL_CLASS = GAN_Wloss
    elif selected_model == "WGAN_GP":
        MODEL_CLASS = GAN_Wloss_GP
    else:
        MODEL_CLASS = CGAN

    if config.resume_id:
        # Download file from the wandb cloud.
        load_checkpoint_from_cloud(checkpoint_path="checkpoint.ckpt")
        extra_trainer_args = {"resume_from_checkpoint": checkpoint_path}
        model = MODEL_CLASS.load_from_checkpoint(checkpoint_path)
    else:
        extra_trainer_args = {}
        model = MODEL_CLASS(config)

    return model, extra_trainer_args


def train(config, run, model, dataModule, extra_trainer_args):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # wandb logger setup
    wandb_logger = WandbLogger(
        experiment=run, log_model=True, save_dir=Path(run.dir).absolute()
    )
    # log config
    wandb.config.update(config)

    checkpoint_path = str(Path(run.dir).absolute() / "checkpoint.ckpt")
    callbacks = [SaveWandbCallback(config.log_interval, checkpoint_path)]

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=wandb.run.dir,
        checkpoint_callback=None,
        **extra_trainer_args,
    )

    # train
    trainer.fit(model, dataModule)


# sample script
def main():
    data_location = "../tests/data"
    config_dict = dict(
        aug_rotation_type="random rotation",
        data_augmentation=True,
        aug_rotation_axis=(0, 1, 0),
        data_location=data_location,
        resume_id="",
        selected_model="GAN",
        log_interval=15,
        log_num_samples=1,
        project_name="tree-gan",
        resolution=32,
        num_classes=0,
        seed=1234,
        epochs=1,
        batch_size=32,
    )
    config = Namespace(**config_dict)
    # adjust parameter datatype
    if config.selected_model in ["VAEGAN", "GAN", "VAE", "WGAN", "WGAN_GP"]:
        config.num_classes = 0

    # run offline
    os.environ["WANDB_MODE"] = "dryrun"
    run, config = init_wandb_run(config, run_dir="../")
    dataModule, config = setup_datamodule(config, run)
    model, extra_trainer_args = setup_model(config, run)

    if torch.cuda.is_available():
        extra_trainer_args["gpus"] = -1

    train(config, run, model, dataModule, extra_trainer_args)


if __name__ == "__main__":
    main()
