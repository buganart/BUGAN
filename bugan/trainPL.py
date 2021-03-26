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
    VAEGAN_Wloss_GP,
    CGAN,
    CVAEGAN,
    CGAN_Wloss_GP,
    CVAEGAN_Wloss_GP,
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
    elif model_name == "VAEGAN_GP":
        MODEL_CLASS = VAEGAN_Wloss_GP
    elif model_name == "CGAN":
        MODEL_CLASS = CGAN
    elif model_name == "CVAEGAN":
        MODEL_CLASS = CVAEGAN
    elif model_name == "CGAN_GP":
        MODEL_CLASS = CGAN_Wloss_GP
    else:
        MODEL_CLASS = CVAEGAN_Wloss_GP
    return MODEL_CLASS


def save_model_args(config, run):
    filepath = str(Path(run.dir).absolute() / "model_args.json")

    # save only the model argument (args in parser)
    parser = get_model_argument_parser(config.selected_model)
    args_keys = vars(parser.parse_args([])).keys()
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


def get_model_argument_parser(model_name):
    MODEL_CLASS = _get_models(model_name)
    return MODEL_CLASS.add_model_specific_args(ArgumentParser())


def _validate_model_config(config):
    parser = get_model_argument_parser(config.selected_model)
    args = parser.parse_args([])
    # check argument and print warning if config arguments not in valid args keys
    args_keys = vars(args).keys()
    valid_config_keys = VALID_CONFIG_KEYWORDS + list(args_keys)
    for k in vars(config):
        if k not in valid_config_keys:
            warnings.warn(f"config argument '{k}' is not one of model arguments.")


def setup_config_arguments(config):
    MODEL_CLASS = _get_models(config.selected_model)
    parser = MODEL_CLASS.add_model_specific_args(ArgumentParser())
    args = parser.parse_args([])
    config = MODEL_CLASS.combine_namespace(args, config)
    return config


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
            extra_trainer_args = {"resume_from_checkpoint": checkpoint_path}
            model = MODEL_CLASS.load_from_checkpoint(checkpoint_path)
            new_ckpt_loaded = True
        except:
            # Download previous successfully loaded checkpoint file
            load_checkpoint_from_cloud(checkpoint_path="checkpoint_prev.ckpt")
            extra_trainer_args = {"resume_from_checkpoint": checkpoint_prev_path}
            model = MODEL_CLASS.load_from_checkpoint(checkpoint_prev_path)

        if new_ckpt_loaded:
            print(
                "checkpoint loaded. Save a copy of successfully loaded checkpoint to the cloud."
            )
            # save successfully loaded checkpoint file as checkpoint_prev.ckpt
            os.rename(checkpoint_path, checkpoint_prev_path)
            save_checkpoint_to_cloud(checkpoint_prev_path)

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
    # update config by filling in missing model argument by default values
    config = setup_config_arguments(config)
    # log config
    wandb.config.update(config)
    save_model_args(config, run)
    pprint.pprint(vars(config))

    checkpoint_path = str(Path(run.dir).absolute())
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
    if config.selected_model in [
        "VAEGAN",
        "GAN",
        "VAE",
        "WGAN",
        "WGAN_GP",
        "VAEGAN_GP",
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
