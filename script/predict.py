import os
import io
import sys
import subprocess
import time
import argparse
import json

import numpy as np
import torch
import wandb
from pathlib import Path

from bugan.trainPL import (
    init_wandb_run,
    setup_model,
    get_resume_run_config,
)
from bugan.functionsPL import netarray2mesh

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


def load_wandb_run(run_id):
    api = wandb.Api()
    try:
        project_name = "tree-gan"
        run = api.run(f"bugan/{project_name}/{run_id}")
    except Exception as e:
        print(e)
        print("set project_name to tree-gan Failed. Try handtool-gan")
        project_name = "handtool-gan"
        run = api.run(f"bugan/{project_name}/{run_id}")
    return run


# generateMesh
def generateMeshFromModel(model, z, class_index=None):
    test_num_samples = z.shape[0]
    # get generator
    if hasattr(model, "vae"):
        generator = model.vae.vae_decoder
        if class_index is not None:
            embedding_fn = model.vae.embedding
    else:
        generator = model.generator
        if class_index is not None:
            embedding_fn = model.embedding

    z = torch.tensor(z).type_as(generator.gen_fc.weight)
    if class_index is not None:
        if isinstance(class_index, int):
            # turn class vector the same device as z, but with dtype Long
            c = torch.ones(test_num_samples) * class_index
            c = c.type_as(z).to(torch.int64)

            # combine z and c
            z = model.merge_latent_and_class_vector(
                z,
                c,
                model.config.num_classes,
                embedding_fn=embedding_fn,
            )
        else:
            # assume class_index are processed class_vectors
            class_vectors = torch.tensor(class_index).type_as(z)

            # merge with z to be generator input
            z = torch.cat((z, class_vectors), 1)

    generated_tree = generator(z)[:, 0, :, :, :]
    generated_tree = generated_tree.detach().cpu().numpy()

    return generated_tree


def generateFromCheckpoint(
    selected_model,
    ckpt_filePath,
    out_dir,
    config,
    class_index=None,
    num_samples=1,
    package_rev_number=None,
    latent=False,
    output_file_name_dict={},
):

    try:
        # restore bugan version
        install_bugan_package(rev_number=package_rev_number)
        from bugan.trainPL import _get_models

        MODEL_CLASS = _get_models(selected_model)
        # try to load model with checkpoint.ckpt
        if config:
            model = MODEL_CLASS.load_from_checkpoint(ckpt_filePath, config=config)
        else:
            model = MODEL_CLASS.load_from_checkpoint(ckpt_filePath)
    except Exception as e:
        print(e)
        print(
            "resume model from previous bugan package rev_number failed. try the newest bugan package"
        )
        # try newest bugan version
        install_bugan_package()
        from bugan.trainPL import _get_models

        MODEL_CLASS = _get_models(selected_model)
        # try to load model with checkpoint_prev.ckpt
        if config:
            model = MODEL_CLASS.load_from_checkpoint(ckpt_filePath, config=config)
        else:
            model = MODEL_CLASS.load_from_checkpoint(ckpt_filePath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("device:", device)
    model = model.to(device)

    if latent or isinstance(class_index, list):

        # generate latent vectors in a line
        z_size = config.z_size
        latent_1 = np.ones(z_size)
        if latent:
            latent_2 = np.ones(z_size) * -1
        else:
            latent_2 = latent_1
        latent_vectors = np.linspace(latent_1, latent_2, num=num_samples)

        if isinstance(class_index, list):
            # process class_index into class_vectors line(assume class_index in format [c1,c2])
            if hasattr(model, "vae"):
                embedding_fn = model.vae.embedding
            else:
                embedding_fn = model.embedding
            c_start, c_end = class_index
            c_start = embedding_fn(torch.tensor(c_start)).detach().cpu().numpy()
            c_end = embedding_fn(torch.tensor(c_end)).detach().cpu().numpy()

            class_vectors = np.linspace(c_start, c_end, num=num_samples)

        # generate loop
        meshes = []
        batch_size = 2
        loops = int(np.ceil(num_samples / batch_size))
        for i in range(loops):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > num_samples:
                end = num_samples
            latent_vector = latent_vectors[start:end]
            try:
                if isinstance(class_index, int):
                    mesh = generateMeshFromModel(
                        model, latent_vector, class_index=class_index
                    )
                else:
                    class_vector = class_vectors[start:end]
                    mesh = generateMeshFromModel(
                        model, latent_vector, class_index=class_vector
                    )
            except Exception as e:
                print(e)
                print(
                    "try generate with class_index failed. Try generate without class_index"
                )
                mesh = generateMeshFromModel(model, latent_vector, class_index=None)
            meshes.append(mesh)
        meshes = np.concatenate(meshes)
    else:
        class_index = int(class_index)
        # random generate mesh
        try:
            # assume conditional model
            mesh = model.generate_tree(c=class_index, num_trees=num_samples)
        except Exception as e:
            print(e)
            print("generate with class label does not work. Now generate without label")
            # assume unconditional model
            mesh = model.generate_tree(num_trees=num_samples)

    print(num_samples, " objects are generated, processing objects to json......")
    save_filename_header = ""
    for k, v in output_file_name_dict.items():
        save_filename_header = save_filename_header + f"_{str(k)}_{str(v)}"

    for i in range(num_samples):
        sample_tree_bool_array = mesh[i] > 0
        voxelmesh = netarray2mesh(sample_tree_bool_array)

        save_filename = f"sample_{i}{save_filename_header}.obj"
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
def generateMesh_local(
    ckpt_dir, out_dir, num_samples=1, rev_number=None, class_index=0, latent=False
):
    ckpt_filePath = ckpt_dir / "checkpoint.ckpt"
    config_filePath = ckpt_dir / "model_args.json"
    # selected_model = "VAEGAN"
    # rev_number = "ed8b9fb"
    # class_index = None

    # config
    try:
        with open(config_filePath, "r") as fp:
            config = json.load(fp)
            config = argparse.Namespace(**config)

        print("loaded model config:", config)
    except:
        config = None
        print(f"{ckpt_dir} not found! use checkpoint stored config.")

    global message_steptime
    message_steptime = []

    message = "starting loading models...."
    print_time_message(message, refresh_time=True)

    # try to load model with latest ckpt
    try:
        selected_model = "VAEGAN"
        generateFromCheckpoint(
            selected_model,
            ckpt_filePath,
            out_dir,
            config,
            class_index=class_index,
            num_samples=num_samples,
            package_rev_number=rev_number,
            latent=latent,
        )
    except Exception as e:
        print(e)
        print("model not set, VAEGAN does not work. Try CVAEGAN....")
        selected_model = "CVAEGAN"
        generateFromCheckpoint(
            selected_model,
            ckpt_filePath,
            out_dir,
            config,
            class_index=class_index,
            num_samples=num_samples,
            package_rev_number=rev_number,
            latent=latent,
        )

    message = "finish generate mesh, time: "
    print_time_message(message)


def generateMesh_run(
    run_id, out_dir, num_samples=1, rev_number=None, class_index=0, latent=False
):

    run = load_wandb_run(run_id)

    config = argparse.Namespace(**run.config)
    # load selected_model, rev_number in the config
    if hasattr(config, "selected_model"):
        selected_model = config.selected_model
    if hasattr(config, "rev_number"):
        rev_number = config.rev_number

    try:
        ckpt_file = run.file("checkpoint.ckpt").download(replace=True)
        generateFromCheckpoint(
            selected_model,
            ckpt_file.name,
            out_dir,
            config,
            class_index=class_index,
            num_samples=num_samples,
            package_rev_number=rev_number,
            latent=latent,
        )
    except Exception as e:
        print(e)
        print("loading from checkpoint.ckpt failed. Try checkpoint_prev.ckpt")
        ckpt_file = run.file("checkpoint_prev.ckpt").download(replace=True)
        generateFromCheckpoint(
            selected_model,
            ckpt_file.name,
            out_dir,
            config,
            class_index=class_index,
            num_samples=num_samples,
            package_rev_number=rev_number,
            latent=latent,
        )


def generateMesh_runHistory(
    run_id,
    num_history_ckpt,
    out_dir,
    num_samples=1,
    rev_number=None,
    class_index=0,
    latent=False,
):

    run = load_wandb_run(run_id)

    config = argparse.Namespace(**run.config)
    # load selected_model, rev_number in the config
    if hasattr(config, "selected_model"):
        selected_model = config.selected_model
    if hasattr(config, "rev_number"):
        rev_number = config.rev_number

    # find necessary checkpoint file
    epoch_list = []
    epoch_file_dict = {}
    for file in run.files():
        filename = file.name
        if not ".ckpt" in filename:
            continue
        if (filename == "checkpoint.ckpt") or (filename == "checkpoint_prev.ckpt"):
            continue
        file_epoch = str((filename.split("_")[1]).split(".")[0])
        epoch_list.append(int(file_epoch))
        epoch_file_dict[file_epoch] = file

    epoch_list = sorted(epoch_list)
    if len(epoch_list) < num_history_ckpt:
        num_history_ckpt = len(epoch_list)
    print(f"select {num_history_ckpt} out of {len(epoch_list)} checkpoints......")
    selected_epoch_index = [
        int(i / (num_history_ckpt - 1) * (len(epoch_list) - 1) + 0.5)
        for i in range(num_history_ckpt)
    ]

    # download checkpoint and generate mesh
    for checkpoint_epoch_index in selected_epoch_index:
        file_epoch = str(epoch_list[checkpoint_epoch_index])
        print(f"generate mesh for epoch {file_epoch}......")
        try:
            ckpt_file = epoch_file_dict[file_epoch]
            ckpt_file.download(replace=True)
            generateFromCheckpoint(
                selected_model,
                ckpt_file.name,
                out_dir,
                config,
                class_index=class_index,
                num_samples=num_samples,
                package_rev_number=rev_number,
                latent=latent,
                output_file_name_dict={"epoch": file_epoch},
            )
        except Exception as e:
            print(e)
            print(f"generate mesh for epoch {file_epoch} FAILED !!!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--num_history_ckpt", type=int, default=-1)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint/")
    parser.add_argument("--out_dir", type=str, default="./output/")
    parser.add_argument("--rev_number", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--class_index", type=int, default=0)
    parser.add_argument("--class_index2", type=int, default=-1)
    parser.add_argument("--latent", type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    install_bugan_package()

    num_samples = args.num_samples
    rev_number = args.rev_number
    class_index = args.class_index
    class_index2 = args.class_index2
    if class_index2 < 0:
        class_index2 = class_index
    else:
        class_index = [class_index, class_index2]

    latent = args.latent

    # check if wandb id
    run_id = args.run_id
    ckpt_dir = args.ckpt_dir
    num_history_ckpt = args.num_history_ckpt
    if run_id:
        if num_history_ckpt > 0:
            generateMesh_runHistory(
                run_id,
                num_history_ckpt,
                out_dir,
                num_samples,
                rev_number,
                class_index,
                latent,
            )
        else:
            generateMesh_run(
                run_id, out_dir, num_samples, rev_number, class_index, latent
            )
    else:
        ckpt_dir = Path(ckpt_dir)
        generateMesh_local(
            ckpt_dir, out_dir, num_samples, rev_number, class_index, latent
        )
