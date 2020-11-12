# BUGAN

## Installation

    pip install -e .


## Development

Install package and development tools

    pip install -r requirements.txt

Install pre-commit hooks

    pre-commit install

This will automatically run the formatter before committing in the future.

To run the hooks (for now just formatting) manually

    pre-commit run

Or to run it on all files

    pre-commit run --all-files

A convenient way to keep the code formatted is to configure the editor/IDE
to format with `black` on save.

Run the tests

    pytest

Or with a file watcher to re-run when saving files

    ptw

## Training

    wandb login  # Only if not yet logged in.
    ./train.py --data-path my-data-path
