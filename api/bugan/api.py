import os
import io
import sys
import subprocess

import click
import time

# import torch
from flask import Flask, jsonify, request
import requests

from example import generated_obj

from argparse import Namespace


from bugan.trainPL import (
    init_wandb_run,
    setup_model,
    get_resume_run_config,
    _get_models,
)
from bugan.functionsPL import netarray2mesh, eval_cluster
import numpy as np

import torch
import wandb
from pathlib import Path


app = Flask(__name__)
app.config["SERVER_NAME"] = os.environ.get("SERVER_NAME")

BUNNY_URL = "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"

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

preset_models = {
    "double_trunk_1": ["vtcf6k3t", 0],
    "double_trunk_2": ["vtcf6k3t", 0],
    "formal_upright_1": ["vtcf6k3t", 0],
    "formal_upright_2": ["vtcf6k3t", 0],
    "friedrich_1": ["vtcf6k3t", 0],
    "friedrich_2": ["vtcf6k3t", 0],
    "friedrich_3": ["1v3odhkm", 2],
    "group_1": ["1fj7x4dk", 4],
    "group_2": ["1fj7x4dk", 1],
    "informal_upright_1": ["1fj7x4dk", 2],
    "informal_upright_2": ["vtcf6k3t", 0],
    "leaning_1": ["vtcf6k3t", 0],
    "leaning_2": ["vtcf6k3t", 2],
    "mustard_reaching_1": ["29k3qjns", 4],
    "mustard_sapling_1": ["vtcf6k3t", 0],
    "mustard_sapling_2": ["1fj7x4dk", 3],
    "pn_banyan_1": ["1v3odhkm", 3],
    "pn_maple_1": ["1fj7x4dk", 0],
    "pn_old_1": ["vtcf6k3t", 0],
    "pn_pine_1": ["29k3qjns", 2],
    "pn_pine_2": ["1v3odhkm", 0],
    "pn_pine_3": ["29k3qjns", 0],
    "pn_pine_4": ["29k3qjns", 1],
    "pn_pine_5": ["vtcf6k3t", 0],
    "pn_pine_6": ["vtcf6k3t", 4],
    "pn_tall_straight": ["29k3qjns", 3],
    "pn_tall_straight_old": ["1v3odhkm", 4],
    "raft_1": ["vtcf6k3t", 0],
    "raft_2": ["1v3odhkm", 1],
    "semi_cascade_1": ["vtcf6k3t", 3],
    "semi_cascade_2": ["vtcf6k3t", 1],
    "sept_chen_lin_1": ["vtcf6k3t", 0],
    "sept_chen_lin_2": ["vtcf6k3t", 0],
    "sept_chen_lin_3": ["vtcf6k3t", 0],
    "sept_constable_1": ["vtcf6k3t", 0],
    "sept_friedrich_4": ["vtcf6k3t", 0],
    "sept_friedrich_5": ["vtcf6k3t", 0],
    "sept_holten_a": ["vtcf6k3t", 0],
    "sept_holten_b": ["vtcf6k3t", 0],
    "sept_holten_c": ["vtcf6k3t", 0],
    "sept_holten_d": ["vtcf6k3t", 0],
    "sept_holten_e": ["vtcf6k3t", 0],
    "sept_holten_f": ["vtcf6k3t", 0],
    "sept_holten_g": ["vtcf6k3t", 0],
    "sept_holten_h": ["vtcf6k3t", 0],
    "sept_holten_i": ["vtcf6k3t", 0],
    "sept_holten_j": ["vtcf6k3t", 0],
    "sept_holten_k": ["vtcf6k3t", 0],
    "sept_holten_l": ["vtcf6k3t", 0],
    "sept_holten_m": ["vtcf6k3t", 0],
    "sept_holten_n": ["vtcf6k3t", 0],
    "sept_holten_o": ["vtcf6k3t", 0],
    "sept_holten_p": ["vtcf6k3t", 0],
    "sept_holten_q": ["vtcf6k3t", 0],
    "sept_holten_r": ["vtcf6k3t", 0],
    "sept_holten_s": ["vtcf6k3t", 0],
    "sept_holten_t": ["vtcf6k3t", 0],
    "sept_holten_u": ["vtcf6k3t", 0],
    "sept_holten_v": ["vtcf6k3t", 0],
    "sept_holten_w": ["vtcf6k3t", 0],
    "sept_holten_x": ["vtcf6k3t", 0],
    "sept_holten_y": ["vtcf6k3t", 0],
    "sept_holten_z": ["vtcf6k3t", 0],
    "sept_mondrian_1": ["vtcf6k3t", 0],
    "sept_mondrian_2": ["vtcf6k3t", 0],
    "sept_mondrian_3": ["vtcf6k3t", 0],
    "sept_schiele_1": ["vtcf6k3t", 0],
    "sept_schiele_2": ["vtcf6k3t", 0],
    "sept_schiele_3": ["vtcf6k3t", 0],
    "windswept_1": ["vtcf6k3t", 0],
    "windswept_2": ["vtcf6k3t", 0],
    "zan_gentlemen_1": ["vtcf6k3t", 0],
    "zan_gentlemen_2": ["vtcf6k3t", 0],
    "zan_gentlemen_3": ["vtcf6k3t", 0],
    "zan_gentlemen_4": ["vtcf6k3t", 0],
    "zan_gentlemen_5": ["vtcf6k3t", 0],
}


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


# post processing array
def cluster_in_sphere(voxel_index_list, center, radius):
    center = np.array(center)
    for v in voxel_index_list:
        v = np.array(v)
        dist = np.linalg.norm(v - center)
        if dist < radius:
            return True
    return False


def post_process_array(boolarray, point_threshold, radius):
    # print("Post-processing is True")
    boolarray = boolarray > 0
    cluster = eval_cluster(boolarray)

    # post process
    process_cluster = []
    for l in cluster:
        l = list(l)
        if len(l) < point_threshold:
            continue
        if not cluster_in_sphere(l, np.array(boolarray.shape) / 2, radius):
            continue
        process_cluster.append(l)

    # point form back to array form
    processed_tree = np.zeros_like(boolarray)
    for c in process_cluster:
        for index in c:
            i, j, k = index
            processed_tree[i, j, k] = 1
    return processed_tree


def generateFromCheckpoint(
    config,
    selected_model,
    ckpt_filePath,
    class_index=None,
    num_samples=1,
    package_rev_number=None,
    point_threshold=None,
    radius=None,
):
    MODEL_CLASS = _get_models(selected_model)
    config.batch_size = 2

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
    returnMeshes = []
    for i in range(num_samples):
        sample_tree_bool_array = mesh[i] > 0
        if point_threshold and radius:
            sample_tree_bool_array = post_process_array(
                sample_tree_bool_array, point_threshold, radius
            )
        voxelmesh = netarray2mesh(sample_tree_bool_array)

        # output as json
        voxelmeshfile = voxelmesh.export(file_type="obj")
        returnMeshes.append(io.StringIO(voxelmeshfile).getvalue())

        # store output as files
        # save_filename = f"sample_{i}.obj"
        # export_path = Path("./") / save_filename
        # voxelmesh.export(file_obj=export_path, file_type="obj")
    return returnMeshes


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


@app.route("/generate", methods=["post"])
def generate():
    req = request.get_json(force=True)
    url = req.get("url", None)

    if url:
        mesh = requests.get(url).text
    else:
        mesh = generated_obj

    return jsonify(mesh=mesh)


# tree classes
@app.route("/getTreeClasses", methods=["post"])
def getTreeClasses():
    class_list = list(preset_models.keys())
    return jsonify(class_list=class_list)


# tree classes
@app.route("/clear", methods=["post"])
def clear():
    ckptfile_list = Path(ckpt_dir).rglob("*.ckpt")
    for path in ckptfile_list:
        os.remove(path)
    return "success"


# generate mesh given (run_id, num_samples, class_index)
# or
# generate mesh given (class_name, num_samples)
@app.route("/generateMesh", methods=["post"])
def generateMesh():
    global message_steptime
    message_steptime = []

    req = request.get_json(force=True)

    run_id = req.get("run_id", None)
    num_samples = int(req.get("num_samples", 1))
    class_index = req.get("class_index", None)
    class_name = req.get("class_name", None)
    point_threshold = req.get("point_threshold", None)
    radius = req.get("radius", None)

    print("req:", req)
    if class_name is not None:
        # convert class_name to run_id
        run_id, class_index = preset_models[class_name]

    if class_index is not None:
        class_index = int(class_index)

    if point_threshold and radius:
        print(f"point_threshold and radius are set. Post-processing is True")

    if not point_threshold and radius:
        point_threshold = 50
        print(
            f"radius is set. Post-processing is True and now set point_threshold to {point_threshold}"
        )

    if point_threshold and not radius:
        radius = 28
        print(
            f"point_threshold is set. Post-processing is True and now set radius to {radius}"
        )

    generateMesh_idList, _ = search_local_checkpoint(ckpt_dir)
    print("stored ckpt id:", generateMesh_idList)
    if run_id:
        message = "starting loading models...."
        print_time_message(message, refresh_time=True)

        try:
            config = get_resume_run_config("handtool-gan", run_id)
        except:
            config = get_resume_run_config("tree-gan", run_id)

        message = "finish loading config setting, time: "
        print_time_message(message)

        filePath = "./checkpoint/" + str(run_id) + "_" + "checkpoint.ckpt"

        api = wandb.Api()
        run = api.run(f"bugan/{config.project_name}/{run_id}")

        message = "finish restoring wandb run environment, time: "
        print_time_message(message)

        if run_id not in generateMesh_idList:
            # downloaded file will be in "./"
            file = run.file("checkpoint.ckpt").download(replace=True)
            file.close()
            # change filename by adding run_id on it
            os.rename("./checkpoint.ckpt", filePath)

            # manage generateMesh_idList
            generateMesh_idList.append(run_id)
            if len(generateMesh_idList) > 10:
                old_id = generateMesh_idList.pop(0)
                # remove the old checkpoint file
                old_filePath = "./checkpoint/" + str(old_id) + "_" + "checkpoint.ckpt"
                os.remove(old_filePath)

            message = "finish restoring checkpoint file, time: "
            print_time_message(message)

        try:
            # try to load model with latest ckpt
            returnMeshes = generateFromCheckpoint(
                config,
                config.selected_model,
                filePath,
                class_index=class_index,
                num_samples=num_samples,
                package_rev_number=config.rev_number,
                point_threshold=point_threshold,
                radius=radius,
            )
        except Exception as e:
            print(e)
            print("loading from checkpoint.ckpt failed. Try checkpoint_prev.ckpt")
            # remove failed checkpoint
            os.remove(filePath)
            # downloaded checkpoint_prev
            file = run.file("checkpoint_prev.ckpt").download(replace=True)
            file.close()
            # change filename by adding run_id on it
            os.rename("./checkpoint_prev.ckpt", filePath)

            message = "finish restoring prev checkpoint file, time: "
            print_time_message(message)

            # try to load model with prev ckpt
            returnMeshes = generateFromCheckpoint(
                config,
                config.selected_model,
                filePath,
                class_index=class_index,
                num_samples=num_samples,
                package_rev_number=config.rev_number,
                point_threshold=point_threshold,
                radius=radius,
            )

        message = "finish generate mesh, time: "
        print_time_message(message)

        print("=== Time summary ===")
        for m, t in message_steptime:
            print(m, t)
        return jsonify(mesh=returnMeshes)
    else:
        # return empty response: 204 No Content?
        return ("", 204)


# generate mesh given (run_id, class_index, num_samples, num_selected_checkpoint)
# or
# generate mesh given (class_name, num_samples, num_selected_checkpoint)
@app.route("/generateMeshHistory", methods=["post"])
def generateMeshHistory():
    global message_steptime
    message_steptime = []

    req = request.get_json(force=True)

    run_id = req.get("run_id", None)
    num_samples = int(req.get("num_samples", 1))
    class_index = req.get("class_index", None)
    num_selected_checkpoint = int(req.get("num_selected_checkpoint", 4))
    class_name = req.get("class_name", None)

    if class_name is not None:
        # convert class_name to run_id
        run_id, class_index = preset_models[class_name]

    if class_index is not None:
        class_index = int(class_index)

    print("req:", req)
    _, generateMesh_idHistoryDict = search_local_checkpoint(ckpt_dir)
    print("stored history:", generateMesh_idHistoryDict)
    if run_id:
        current_time = time.time()
        message = "starting loading models...."
        print_time_message(message, refresh_time=True)

        try:
            config = get_resume_run_config("handtool-gan", run_id)
        except:
            config = get_resume_run_config("tree-gan", run_id)

        message = "finish loading config setting, time: "
        print_time_message(message)

        api = wandb.Api()
        run = api.run(f"bugan/{config.project_name}/{run_id}")

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
        if len(epoch_list) < num_selected_checkpoint:
            num_selected_checkpoint = len(epoch_list)
        selected_epoch_index = [
            int(i / (num_selected_checkpoint - 1) * (len(epoch_list) - 1) + 0.5)
            for i in range(num_selected_checkpoint)
        ]

        message = "finish finding necessary checkpoint file, time: "
        print_time_message(message)
        # downloaded file will be in "./"
        returnMeshesAll = {}
        for checkpoint_epoch_index in selected_epoch_index:
            file_epoch = str(epoch_list[checkpoint_epoch_index])
            print(f"generate mesh for epoch {file_epoch}......")
            try:
                file = epoch_file_dict[file_epoch]
                filename = file.name
                filePath = f"./checkpoint/{run_id}_checkpoint-{file_epoch}.ckpt"
                if (run_id not in generateMesh_idHistoryDict) or (
                    int(file_epoch) not in generateMesh_idHistoryDict[run_id]
                ):
                    file.download(replace=True)
                    # file.close()
                    # change filename by adding run_id on it
                    os.rename(f"./{filename}", filePath)

                message = f"finish loading checkpoint for epoch {file_epoch}, time: "
                print_time_message(message)

                returnMeshes = generateFromCheckpoint(
                    config.selected_model,
                    filePath,
                    class_index=class_index,
                    num_samples=num_samples,
                    package_rev_number=config.rev_number,
                )
                returnMeshesAll[file_epoch] = returnMeshes

                message = f"finish generate mesh for epoch {file_epoch}, time: "
                print_time_message(message)
            except Exception as e:
                print(e)
                message = f"generate mesh for epoch {file_epoch} FAILED !!!"
                print_time_message(message, refresh_time=True)

        print("=== Time summary ===")
        for m, t in message_steptime:
            print(m, t)
        return jsonify(mesh=returnMeshesAll)
    else:
        # return empty response: 204 No Content?
        return ("", 204)


# @app.route("/version", methods=["GET"])
# def version():
#     with open(VERSION_PATH) as f:
#         return f.read().strip()


@app.route("/status", methods=["GET"])
def status():
    return "ok"


def initialize(checkpoint_dir):

    print("\nThis is a test container, no initialization needed.")
    print("To return a sample mesh for testing: \n")
    print(
        "\tcurl -X POST -H \"Content-Type: application/json\" -d '{}' 127.0.0.1:8080/generate\n"
    )
    print("Or fetch and return a wavefront file from elsewhere, for example:\n")
    print(
        '\tcurl -X POST -H "Content-Type: application/json" -d \'{"url": "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"}\' 127.0.0.1:8080/generate\n'
    )
    print('This returns a json object of the form: {"mesh": WAVEFRONT_LINES},')
    print(
        "where WAVEFRONT_LINES is a string containing the lines of a wavefron file.\n"
    )

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print(f"=> Loading model {checkpoint_dir}")
    # model = load_model(checkpoint_dir).to(device)

    # print(f"=> Loading tokenizer {checkpoint_dir}")
    # tokenizer = load_tokenizer(checkpoint_dir)
    # app.config.update(
    #     dict(
    #         model=model,
    #         tokenizer=tokenizer,
    #         device=device,
    #     )
    # ) .


def setup(cli_checkpoint_dir="./checkpoint"):
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR") or cli_checkpoint_dir
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if not checkpoint_dir:
        raise ValueError("Set --chekpoint-dir or CHECKPOINT_DIR")
    initialize(checkpoint_dir)

    return app


def search_local_checkpoint(path="./"):
    ckptfile_list = Path(path).rglob("*.ckpt")
    # make id list for those checkpoints without epoch number (are latest checkpoint)
    nonHistory_list = [
        str(Path(file).stem)
        for file in ckptfile_list
        if "-" not in str(Path(file).stem)
    ]
    idList = [(filename.split("_"))[0] for filename in nonHistory_list]
    print("recovered run_id:", idList)

    ckptfile_list = Path(path).rglob("*.ckpt")
    # make history dict for those checkpoints with epoch number
    idHistoryDict = {}
    history_list = [
        str(Path(file).stem) for file in ckptfile_list if "-" in str(Path(file).stem)
    ]
    idHistoryList = [(filename.split("_"))[0] for filename in history_list]
    epochHistoryList = sorted(
        [int((filename.split("-"))[1]) for filename in history_list]
    )
    for i in range(len(idHistoryList)):
        idHistory = idHistoryList[i]
        epochHistory = epochHistoryList[i]
        if idHistory in idHistoryDict:
            idHistoryDict[idHistory].append(epochHistory)
        else:
            idHistoryDict[idHistory] = [epochHistory]
    print("recovered run_id history:", idHistoryDict)
    return idList, idHistoryDict


@click.command()
@click.option("--debug", "-d", is_flag=True)
@click.option("--checkpoint-dir", "-cp", default="./checkpoint")
def api_run(debug, checkpoint_dir):
    app = setup(cli_checkpoint_dir=checkpoint_dir)
    global generateMesh_idList, generateMesh_idHistoryDict, ckpt_dir
    ckpt_dir = checkpoint_dir
    generateMesh_idList, generateMesh_idHistoryDict = search_local_checkpoint(ckpt_dir)
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


if __name__ == "__main__":
    api_run()
