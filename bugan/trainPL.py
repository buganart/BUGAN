import io
import os
from io import BytesIO
import sys
import warnings
import json
import zipfile
import trimesh
import numpy as np
from argparse import Namespace, ArgumentParser
import wandb
from pathlib import Path
import pkg_resources
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

VALID_CONFIG_KEYWORDS = [
    "aug_rotation_type",
    "data_augmentation",
    "aug_rotation_axis",
    "data_location",
    "selected_model",
    "project_name",
    "num_classes",
    "seed",
    "epochs",
    "resume_id",
    "dataset",
    "rev_number",
]

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from bugan.functionsPL import (
    save_checkpoint_to_cloud,
    load_checkpoint_from_cloud,
    SaveWandbCallback,
)
from bugan.datamodulePL import DataModule_process
from bugan.modelsPL import (
    VAEGAN,
    VAE_train,
    GAN,
    GAN_Wloss,
    GAN_Wloss_GP,
    CGAN,
    CVAEGAN,
    ZVAEGAN,
)


def _get_models(model_name):
    if model_name == "VAEGAN":
        MODEL_CLASS = VAEGAN
    elif model_name == "GAN":
        MODEL_CLASS = GAN
    elif model_name == "VAE":
        MODEL_CLASS = VAE_train
    elif model_name == "WGAN":
        MODEL_CLASS = GAN_Wloss
    elif model_name == "WGAN_GP":
        MODEL_CLASS = GAN_Wloss_GP
    elif model_name == "CGAN":
        MODEL_CLASS = CGAN
    elif model_name == "CVAEGAN":
        MODEL_CLASS = CVAEGAN
    elif model_name == "ZVAEGAN":
        MODEL_CLASS = ZVAEGAN
    return MODEL_CLASS


def save_model_args(config, run):
    filepath = str(Path(run.dir).absolute() / "model_args.json")

    # save only the model argument (args in parser)
    MODEL_CLASS = _get_models(config.selected_model)
    args = MODEL_CLASS.setup_config_arguments(config)
    args_keys = vars(args).keys()
    config = vars(config)
    config_dict = {}
    for k in args_keys:
        config_dict[k] = config[k]

    with open(filepath, "w") as fp:
        json.dump(config_dict, fp)
    save_checkpoint_to_cloud(filepath)


def load_model_args(filepath):
    with open(filepath, "r") as fp:
        config_dict = json.load(fp)
        return Namespace(**config_dict)
    return None


def _validate_model_config(config):
    MODEL_CLASS = _get_models(config.selected_model)
    args = MODEL_CLASS.setup_config_arguments(config)
    # check argument and print warning if config arguments not in valid args keys
    args_keys = vars(args).keys()
    valid_config_keys = VALID_CONFIG_KEYWORDS + list(args_keys)
    for k in vars(config):
        if k not in valid_config_keys:
            warnings.warn(f"config argument '{k}' is not one of model arguments.")


def setup_config_arguments(config):
    MODEL_CLASS = _get_models(config.selected_model)
    args = MODEL_CLASS.setup_config_arguments(config)
    config = MODEL_CLASS.combine_namespace(args, config)
    config_dict = vars(config)
    cleaned_config = {}
    for k, v in config_dict.items():
        if "/" not in k:
            cleaned_config[k] = v
    cleaned_config = Namespace(**cleaned_config)
    return cleaned_config


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
    # run.tags = run.tags + (selected_model,)
    return run, config


def setup_datamodule(config, tmp_folder="/tmp"):
    if not hasattr(config, "seed"):
        config.seed = 123
    if not hasattr(config, "batch_size"):
        config.batch_size = 4
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset_path = Path(config.data_location)
    dataModule = DataModule_process(config, dataset_path, tmp_folder=tmp_folder)

    print("dataset name: ", config.dataset)
    print("dataset path: ", dataset_path)
    return dataModule


def setup_model(config, run):
    selected_model = config.selected_model
    # model
    MODEL_CLASS = _get_models(selected_model)

    # validate config
    _validate_model_config(config)

    if config.resume_id:
        checkpoint_path = str(Path(run.dir).absolute() / "checkpoint.ckpt")
        checkpoint_prev_path = str(Path(run.dir).absolute() / "checkpoint_prev.ckpt")
        new_ckpt_loaded = False
        try:
            # Download file from the wandb cloud.
            load_checkpoint_from_cloud(checkpoint_path="checkpoint.ckpt")
            model = MODEL_CLASS.load_from_checkpoint(checkpoint_path, config=config)
            new_ckpt_loaded = True
        except:
            # Download previous successfully loaded checkpoint file
            load_checkpoint_from_cloud(checkpoint_path="checkpoint_prev.ckpt")
            model = MODEL_CLASS.load_from_checkpoint(
                checkpoint_prev_path, config=config
            )

        if new_ckpt_loaded:
            print(
                "checkpoint loaded. Save a copy of successfully loaded checkpoint to the cloud."
            )
            # save successfully loaded checkpoint file as checkpoint_prev.ckpt
            os.rename(checkpoint_path, checkpoint_prev_path)
            save_checkpoint_to_cloud(checkpoint_prev_path)

        extra_trainer_args = {"resume_from_checkpoint": checkpoint_prev_path}

    else:
        extra_trainer_args = {}
        model = MODEL_CLASS(config)

    return model, extra_trainer_args


def train(config, run, model, dataModule, extra_trainer_args):
    if not hasattr(config, "seed"):
        config.seed = 123
    if not hasattr(config, "epochs"):
        config.epochs = 1000
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # wandb logger setup
    wandb_logger = WandbLogger(
        experiment=run, log_model=True, save_dir=Path(run.dir).absolute()
    )
    # update config by filling in missing model argument by default values
    config = setup_config_arguments(config)
    # log config
    wandb.config.update(config, allow_val_change=True)
    save_model_args(config, run)
    pprint.pprint(vars(config))

    checkpoint_path = str(Path(run.dir).absolute())
    history_checkpoint_frequency = 0
    if hasattr(config, "history_checkpoint_frequency"):
        if config.history_checkpoint_frequency:
            history_checkpoint_frequency = config.history_checkpoint_frequency

    callbacks = [
        SaveWandbCallback(
            config.log_interval, checkpoint_path, history_checkpoint_frequency
        )
    ]

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
    if config.selected_model in [
        "VAEGAN",
        "GAN",
        "VAE",
        "WGAN",
        "WGAN_GP",
    ]:
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
    dataModule = setup_datamodule(config)
    model, extra_trainer_args = setup_model(config, run)

    if torch.cuda.is_available():
        extra_trainer_args["gpus"] = -1

    train(config, run, model, dataModule, extra_trainer_args)


if __name__ == "__main__":
    main()
