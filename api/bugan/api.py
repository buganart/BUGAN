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
from bugan.functionsPL import netarray2mesh

import torch
import wandb
from pathlib import Path


app = Flask(__name__)
app.config["SERVER_NAME"] = os.environ.get("SERVER_NAME")

BUNNY_URL = "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"


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


@app.route("/generate", methods=["post"])
def generate():
    req = request.get_json(force=True)
    url = req.get("url", None)

    if url:
        mesh = requests.get(url).text
    else:
        mesh = generated_obj

    return jsonify(mesh=mesh)


generateMesh_idList = []


@app.route("/generateMesh", methods=["post"])
def generateMesh():
    message_steptime = []
    req = request.get_json(force=True)

    run_id = req.get("run_id", None)
    num_samples = int(req.get("num_samples", 1))
    print("req:", req)
    if run_id:
        print("starting loading models....")
        current_time = time.time()

        try:
            config = get_resume_run_config("handtool-gan", run_id)
        except:
            config = get_resume_run_config("tree-gan", run_id)

        message = "finish loading config setting, time: "
        step_time = time.time() - current_time
        print(message, step_time)
        message_steptime.append([message, step_time])
        current_time = time.time()

        install_bugan_package(rev_number=config.rev_number)

        message = "finish restoring bugan version for the model, time: "
        step_time = time.time() - current_time
        print(message, step_time)
        message_steptime.append([message, step_time])
        current_time = time.time()

        filePath = "./" + str(run_id) + "_" + "checkpoint.ckpt"

        if run_id not in generateMesh_idList:
            api = wandb.Api()
            run = api.run(f"bugan/{config.project_name}/{run_id}")

            message = "finish restoring wandb run environment, time: "
            step_time = time.time() - current_time
            print(message, step_time)
            message_steptime.append([message, step_time])
            current_time = time.time()

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
                old_filePath = "./" + str(old_id) + "_" + "checkpoint.ckpt"
                os.remove(old_filePath)

            message = "finish restoring checkpoint file, time: "
            step_time = time.time() - current_time
            print(message, step_time)
            message_steptime.append([message, step_time])
            current_time = time.time()

        MODEL_CLASS = _get_models(config.selected_model)
        model = MODEL_CLASS.load_from_checkpoint(filePath)

        message = "finish loading model, time: "
        step_time = time.time() - current_time
        print(message, step_time)
        message_steptime.append([message, step_time])
        current_time = time.time()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        print("device:", device)
        model = model.to(device)
        mesh = model.generate_tree(num_trees=num_samples)

        message = "finish generating 3D objects, time: "
        step_time = time.time() - current_time
        print(message, step_time)
        message_steptime.append([message, step_time])
        current_time = time.time()

        print(num_samples, " objects are generated, processing objects to json......")
        returnMeshes = []
        for i in range(num_samples):
            sample_tree_bool_array = mesh[i] > 0
            voxelmesh = netarray2mesh(sample_tree_bool_array)
            voxelmeshfile = voxelmesh.export(file_type="obj")
            returnMeshes.append(io.StringIO(voxelmeshfile).getvalue())

        message = "finish processing objects to json, time: "
        step_time = time.time() - current_time
        print(message, step_time)
        message_steptime.append([message, step_time])
        current_time = time.time()

        print("=== Time summary ===")
        print("device:", device)
        for m, t in message_steptime:
            print(m, t)
        return jsonify(mesh=returnMeshes)
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


def setup(cli_checkpoint_dir="checkpoint"):
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR") or cli_checkpoint_dir
    if not checkpoint_dir:
        raise ValueError("Set --chekpoint-dir or CHECKPOINT_DIR")
    initialize(checkpoint_dir)
    return app


@click.command()
@click.option("--debug", "-d", is_flag=True)
@click.option("--checkpoint-dir", "-cp", default="checkpoint")
def main(debug, checkpoint_dir):
    app = setup(cli_checkpoint_dir=checkpoint_dir)
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


if __name__ == "__main__":
    main()
