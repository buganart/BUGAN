# BUGAN
![CI](https://github.com/buganart/BUGAN/workflows/CI/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/buganart/BUGAN/branch/master/graph/badge.svg)](https://codecov.io/gh/buganart/BUGAN)

## Dataset

The tree dataset with around 26000 trees in 76 classes are available to [download here](https://drive.google.com/file/d/1_pC6qarczSZk6oiqgmefxH2IuUlU8aLh/view?usp=sharing).

## Colab Notebook

To try the model in this repository, go to **notebook_util** folder.

### Pre-trained Models
To generate meshes from pre-trained model, run notebook **generate_tree(drive).ipynb** and **generate_handtool(drive).ipynb**.
Make sure the Google Account you mount the drive is the same as the one you open the shared folder link in the notebook.

### Train your own model
open the **train.ipynb** to train model in Colab. Modify the parameters in the config panel. 

For example:
Fill in the wandb `project_name` in the format of `{entity}/{project}` (or use default open project `bugan/bu-3dgan-open`), and fill in the `data_location_option` to the folder in your drive containing all `.obj` files.

For modifying advanced parameters, go to **train setup config and package** section to modify parameters list there.

When the training is started, remember to record the wandb run_id (length 8) in the training section. This run_id can be posted in `resume_id` to resume the run, or posted in `id` in the generate notebook for mesh generation.

### Generate mesh from your trained model
open the **generate.ipynb** to train model in Colab. Modify the parameters in the config panel. 

For example:
Put the wandb run_id from the train notebook to the `id` field in the config panel, and fill in the `project_name` in the format of `{entity}/{project}`.

After the notebook finish the run, the generated meshes will be in the path specified in the `export_location`.


## Script

For training using script, go to **script** folder.

As wandb is used in this project, make sure to login to wandb.

    wandb login  # Only if not yet logged in.

### Training

    python script/train.py
    
All the scripts are based on the original **script/train.py**, but with parameter settings changed.

You can run the models with the settings in the scripts, or modify the parameters in the script directly and train.

For wandb, `project_name` should be in the format of `{entity}/{project}` (or use default open project `bugan/bu-3dgan-open`)

Remember to record the wandb run_id (length 8) in the training section, which can be used for resume run in the `resume_id` field, and for mesh generation in the `run_id` field.

### Predict
    
    python script/predict.py --run_id <run_id>

The run_id should come from the model you trained.

Parameters listed in **generate.ipynb** colab notebook can also be used here.

For example: `python script/predict.py --run_id <run_id> --num_samples 2 --post_process True --class_index 1 --latent True`

Note that if `--latent True`, latent space walk is set, and the latent vectors will be sampled evenly from linear space between **1** and **-1**.
