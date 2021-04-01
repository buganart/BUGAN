# BUGAN API

BUGAN API generate meshes based on trained models stored in wandb `bugan` project.

## Start API server

### Recommended
Run the docker container with

    docker run -p 8080:8080 -it buganart/bugan-test-api:0.1.2

For the docker install instruction, see https://docs.docker.com/get-docker/
Install docker based on your OS with default settings.

### Alternative
Clone the repository, and start with the api file
    
    git clone https://github.com/buganart/BUGAN
    cd BUGAN/api/bugan
    python api.py

Please make sure that `flask` and `click` are installed.
This alternative is not supported and may not work. Also, the BUGAN package version in the local machine will be changed when the api server is working.

## API usage

### Requests format
API requests are made in the format below

    curl -X POST -H "Content-Type: application/json" -d <request_content> 127.0.0.1:8080/<function>
    
The `ip_address` should be `127.0.0.1` for windows by default, and `0.0.0.0` for linux by default.

### Return format
The return value of the mesh function request is usually a json object `{"mesh": [mesh1, mesh2, ....]}`,
where `mesh1` is a string containing the lines of a wavefront file:

```
# https://github.com/mikedh/trimesh
v 0.00000000 16.00000000 16.50000000
v 0.00000000 16.00000000 14.50000000
v 0.00000000 15.00000000 16.50000000
...
```

### Function: generateMesh
To return 2 sample meshes based on `run_id` and `class_index` from the wandb `bugan` project:

    curl -X POST -H "Content-Type: application/json" -d '{"run_id" : "46k3x6pu","class_index" : "2", "num_samples" : "2"}' 127.0.0.1:8080/generateMesh

`class_index` can be missing from the request if the model is unconditional.
if `num_samples` is missing, 1 mesh will be returned.

To return 10 sample meshes based on `class_name` (using preset models) from the wandb `bugan` project:

    curl -X POST -H "Content-Type: application/json" -d '{"class_name" : "friedrich_3","num_samples" : "10"}' 127.0.0.1:8080/generateMesh

The API will use stored `run_id` and `class_index` based on `class_name` to generate meshes.

The return value of the mesh function request is a json object `{"mesh": [mesh1, mesh2, ....]}`. See Return format.

### Function: generateMeshHistory
The function `generateMeshHistory` behaves similar to the function `generateMesh`, but instead of only generate meshes from the last model checkpoint, this function will generate meshes for each of the selected history checkpoints that is saved based on the training epochs.

Extra request value `num_selected_checkpoint` to specify the number of history checkpoints for generating meshes. Default 4.

For example, a bugan model is trained for 100 epochs and save history for each 20 epochs, the history checkpoint list is [0, 20, 40, 60, 80, 100].
If `num_selected_checkpoint` is 3, then checkpoints [0, 40, 100] will be selected.

To return 2 sample meshes from 3 history checkpoints based on `class_name` (using preset models) from the wandb `bugan` project:

    curl -X POST -H "Content-Type: application/json" -d '{"class_name" : "friedrich_3","num_samples" : "2", "num_selected_checkpoint" : "3"}' 127.0.0.1:8080/generateMeshHistory

To return 1 sample meshes from 4 history checkpoints based on `run_id` and `class_index` from the wandb `bugan` project
    
    curl -X POST -H "Content-Type: application/json" -d '{"run_id" : "46k3x6pu","class_index" : "2"}' 127.0.0.1:8080/generateMeshHistory

The return value of the mesh function request is a json object `{"mesh": { "0": [mesh1, mesh2, ....], "40": [mesh1, mesh2, ....], ...... }}`, where the index of the `mesh` dictionary is the checkpoint epoch.
For mesh1 format, See Return format.

### Function: getTreeClasses
This function displays all the available tree classes, which are the values of `class_name`.

To return all tree class list:

    curl -X POST -H "Content-Type: application/json" -d '{}' 127.0.0.1:8080/getTreeClasses
    
The return value of request is a json object `{"class_list": ["double_trunk_1","double_trunk_2","formal_upright_1",......]}`

### Function: clear
From the functions above, after downloading the checkpoints to generate trees, the checkpoint will be saved and reused for new requests with the same `run_id`.
This function will remove all the saved checkpoints from the api server.

To remove all checkpoints saved in the api server:
    
    curl -X POST -H "Content-Type: application/json" -d '{}' 127.0.0.1:8080/clear
