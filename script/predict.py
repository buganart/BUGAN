import os
import io
import sys
import subprocess
import time
import argparse
import json

import torch
import wandb
from pathlib import Path

from bugan.trainPL import (
    init_wandb_run,
    setup_model,
    get_resume_run_config,
    _get_models,
)
from bugan.functionsPL import netarray2mesh


global generateMesh_idList
generateMesh_idList = []

global generateMesh_idHistoryDict
generateMesh_idHistoryDict = {}

global ckpt_dir
ckpt_dir = "./checkpoint"

global current_time
current_time = time.time()

global message_steptime
message_steptime = []


def install_bugan_package(rev_number=None):
    if rev_number:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                f"git+https://github.com/buganart/BUGAN.git@{rev_number}#egg=bugan",
            ]
        )
    else:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "git+https://github.com/buganart/BUGAN.git#egg=bugan",
            ]
        )


def generateFromCheckpoint(
    selected_model,
    ckpt_filePath,
    out_dir,
    config,
    class_index=None,
    num_samples=1,
    package_rev_number=None,
):
    MODEL_CLASS = _get_models(selected_model)

    try:
        # restore bugan version
        install_bugan_package(rev_number=package_rev_number)
        # try to load model with checkpoint.ckpt
        model = MODEL_CLASS.load_from_checkpoint(ckpt_filePath, config=config)
    except Exception as e:
        print(e)
        print(
            "resume model from previous bugan package rev_number failed. try the newest bugan package"
        )
        # try newest bugan version
        install_bugan_package()
        # try to load model with checkpoint_prev.ckpt
        model = MODEL_CLASS.load_from_checkpoint(ckpt_filePath, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("device:", device)
    model = model.to(device)
    try:
        # assume conditional model
        mesh = model.generate_tree(c=class_index, num_trees=num_samples)
    except Exception as e:
        print(e)
        print("generate with class label does not work. Now generate without label")
        # assume unconditional model
        mesh = model.generate_tree(num_trees=num_samples)

    print(num_samples, " objects are generated, processing objects to json......")
    for i in range(num_samples):
        sample_tree_bool_array = mesh[i] > 0
        voxelmesh = netarray2mesh(sample_tree_bool_array)

        save_filename = f"sample_{i}.obj"
        export_path = Path(out_dir) / save_filename
        voxelmesh.export(file_obj=export_path, file_type="obj")


def print_time_message(message, refresh_time=False):
    global current_time, message_steptime
    if refresh_time:
        current_time = time.time()
        print(message)
    else:
        step_time = time.time() - current_time
        print(message, step_time)
        message_steptime.append([message, step_time])
        current_time = time.time()


# generate mesh given (out_dir, num_samples)
def generateMesh(out_dir, num_samples):
    filePath = "./checkpoint/checkpoint.ckpt"
    selected_model = "VAEGAN"
    rev_number = "ed8b9fb"
    class_index = None

    # config
    with open("./checkpoint/model_args.json", "r") as fp:
        config = json.load(fp)
        config = argparse.Namespace(**config)

    print("loaded model config:", config)

    global message_steptime
    message_steptime = []

    message = "starting loading models...."
    print_time_message(message, refresh_time=True)

    # try to load model with latest ckpt
    generateFromCheckpoint(
        selected_model,
        filePath,
        out_dir,
        config,
        class_index=class_index,
        num_samples=num_samples,
        package_rev_number=rev_number,
    )

    message = "finish generate mesh, time: "
    print_time_message(message)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./output/")
    parser.add_argument("--num_samples", type=int, default=2)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    generateMesh(out_dir, args.num_samples)
