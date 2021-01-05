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
import pkg_resources

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


def get_resume_run_config(project_name, resume_id):
    # all config will be replaced by the stored one in wandb
    api = wandb.Api()
    previous_run = api.run(f"bugan/{project_name}/{resume_id}")
    config = Namespace(**previous_run.config)
    return config


def get_bugan_package_revision_number():
    version_str = pkg_resources.get_distribution("bugan").version
    rev_number = (version_str.split("+g")[1]).split(".")[0]
    return rev_number


def init_wandb_run(config, run_dir="./", mode="run"):
    resume_id = config.resume_id
    project_name = config.project_name
    selected_model = config.selected_model
    entity = "bugan"
    run_dir = Path(run_dir).absolute()

    if resume_id:
        run_id = resume_id
    else:
        run_id = wandb.util.generate_id()

    run = wandb.init(
        project=project_name,
        id=run_id,
        entity=entity,
        resume=True,
        dir=run_dir,
        group=selected_model,
        mode=mode,
    )

    print("run id: " + str(wandb.run.id))
    print("run name: " + str(wandb.run.name))
    wandb.watch_called = False
    run.tags.append(selected_model)
    return run, config


def setup_datamodule(config, run, tmp_folder="/tmp"):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset_path = Path(config.data_location)
    dataModule = DataModule_process(config, run, dataset_path, tmp_folder=tmp_folder)

    print("dataset name: ", config.dataset)
    print("dataset path: ", dataset_path)
    return dataModule


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

    dataset_path = Path(config.data_location)
    if str(config.data_location).endswith(".zip"):
        config.dataset = dataset_path.stem
    else:
        config.dataset = "dataset_array_custom"
    # adjust parameter datatype
    if config.selected_model in ["VAEGAN", "GAN", "VAE", "WGAN", "WGAN_GP"]:
        config.num_classes = 0

    # run offline
    os.environ["WANDB_MODE"] = "dryrun"

    # get previous config if resume run
    if config.resume_id:
        project_name = config.project_name
        resume_id = config.resume_id
        prev_config = get_resume_run_config(project_name, resume_id)
        # replace config with prev_config
        config = vars(config)
        config.update(vars(prev_config))
        config = Namespace(**config)

    # write bugan package revision number to bugan
    config.rev_number = get_bugan_package_revision_number()

    run, config = init_wandb_run(config, run_dir="../")
    dataModule = setup_datamodule(config, run)
    model, extra_trainer_args = setup_model(config, run)

    if torch.cuda.is_available():
        extra_trainer_args["gpus"] = -1

    train(config, run, model, dataModule, extra_trainer_args)


if __name__ == "__main__":
    main()
