import os

import click

# import torch
from flask import Flask, jsonify, request
import requests

from .example import generated_obj


app = Flask(__name__)
app.config["SERVER_NAME"] = os.environ.get("SERVER_NAME")

BUNNY_URL = "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"


@app.route("/generate", methods=["post"])
def generate():

    req = request.get_json(force=True)

    url = req.get("url", None)

    if url:
        mesh = requests.get(url).text
    else:
        mesh = generated_obj

    return jsonify(mesh=mesh)


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
