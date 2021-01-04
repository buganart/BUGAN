#!/usr/bin/env python
import os
import torch
import wandb
from argparse import Namespace
from pathlib import Path

from bugan.trainPL import (
    get_resume_run_config,
    get_bugan_package_revision_number,
    init_wandb_run,
    setup_datamodule,
    setup_model,
    train,
)


data_path = "./tests/data/"
config_dict = dict(
    aug_rotation_type="random rotation",
    data_augmentation=True,
    aug_rotation_axis=(0, 1, 0),
    data_location=data_path,
    resume_id="",
    selected_model="GAN",
    log_interval=1,
    log_num_samples=1,
    project_name="tree-gan",
    resolution=32,
    num_classes=0,
    seed=1234,
    epochs=1,
    batch_size=32,
    gen_num_layer_unit=[1, 1, 1, 1],
    dis_num_layer_unit=[1, 1, 1, 1],
)
config = Namespace(**config_dict)
dataset_path = Path(config.data_location)
if str(config.data_location).endswith(".zip"):
    config.dataset = dataset_path.stem
else:
    config.dataset = "dataset_array_custom"

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
