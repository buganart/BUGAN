import io
import os
from pathlib import Path

import zipfile
from io import BytesIO

import trimesh
import numpy as np
import tqdm
import torch
import wandb
from PIL import Image

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pytorch_lightning.callbacks.base import Callback
from disjoint_set import DisjointSet


#####
#   functions/callbacks for script, colab
#####

# callback for pl.Trainer() to save_checkpoint() in log_interval
class SaveWandbCallback(Callback):
    def __init__(self, log_interval, save_model_path, history_checkpoint_frequency=0):
        super().__init__()
        self.log_interval = log_interval
        self.save_model_path = Path(save_model_path)
        self.history_checkpoint_frequency = history_checkpoint_frequency

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if trainer.current_epoch % self.log_interval == 0:
            # log
            model_file_path = str(self.save_model_path / "checkpoint.ckpt")
            trainer.save_checkpoint(model_file_path)
            save_checkpoint_to_cloud(model_file_path)
            # record extra checkpoints for history record
            if self.history_checkpoint_frequency:
                if (
                    trainer.current_epoch
                    % (self.log_interval * self.history_checkpoint_frequency)
                    == 0
                ):
                    new_model_path = str(
                        self.save_model_path
                        / (f"checkpoint_{trainer.current_epoch}.ckpt")
                    )
                    trainer.save_checkpoint(new_model_path)
                    save_checkpoint_to_cloud(new_model_path)


# function to save/load files from wandb
def save_checkpoint_to_cloud(checkpoint_path):
    wandb.save(checkpoint_path)


def load_checkpoint_from_cloud(checkpoint_path="model_dict.pth"):
    checkpoint_file = wandb.restore(checkpoint_path)
    return checkpoint_file.name


#####
#   functions for wandbLog in modelPL
#   calculate_log_media_stat() is called by BaseModel on_train_epoch_end()
#####

# convert boolean voxel array (shape: H,W,D) to voxel coordinate list (shape: num_points, 3)
def netarray2indices(boolarray):
    coord_list = []
    if len(boolarray.shape) == 5:
        boolarray = boolarray[0][0]
    x, y, z = boolarray.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if boolarray[i, j, k]:
                    coord_list.append([i, j, k])
    # print(len(coord_list))
    if len(coord_list) == 0:
        return np.array(
            [[0, 0, 0]]
        )  # return at least one point to prevent wandb 3dobject error
    return np.array(coord_list)


# convert boolean voxel array (shape: H,W,D)
# to trimesh VoxelGrid.marching_cubes, the mesh for visualization
def netarray2mesh(array, threshold=0):
    if len(array.shape) != 3:
        raise Exception("netarray2mesh: input array should be 3d")

    # convert to bool dtype
    array = array > threshold
    # array all zero gives error
    if np.sum(array) == 0:
        array[0, 0, 0] = True
    voxelmesh = trimesh.voxel.base.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(array)
    ).marching_cubes
    return voxelmesh


# use DisjointSet to calculate number of voxel clusters in a mesh voxelgrid
# more than 1 cluster means that the mesh may have outliers / floating voxel
# the distance function is p-inf. (see infinity norm/maximum norm)
def eval_count_cluster(boolarray):
    def nearby_voxels(boolarray, i, j, k):
        bound = boolarray.shape
        low_i, high_i = np.clip([i - 1, i + 1], 0, bound[0] - 1)
        low_j, high_j = np.clip([j - 1, j + 1], 0, bound[1] - 1)
        low_k, high_k = np.clip([k - 1, k + 1], 0, bound[2] - 1)

        retval = []
        for x in range(low_i, high_i + 1):
            for y in range(low_j, high_j + 1):
                for z in range(low_k, high_k + 1):
                    if boolarray[x, y, z]:
                        # voxel exists
                        retval.append((x, y, z))
        # remove the selected vox itself
        retval.remove((i, j, k))
        return retval

    ds = DisjointSet()
    for i in range(boolarray.shape[0]):
        for j in range(boolarray.shape[1]):
            for k in range(boolarray.shape[2]):
                vox = boolarray[i, j, k]
                if vox:
                    # voxel exists in coord (i,j,k)
                    nearby_vox = nearby_voxels(boolarray, i, j, k)
                    for v in nearby_vox:
                        ds.union(v, (i, j, k))
    return len(list(ds.itersets()))


# render image (PIL) from voxelmesh 3d object
# using trimesh save_image() function, need a default display/renderer
def mesh2wandbImage(voxelmesh, wandb_format=True):
    scene = voxelmesh.scene()
    try:
        png = scene.save_image(
            resolution=[600, 600],
        )
    except:
        print(
            "NoSuchDisplayException. Renderer not found! Please check configuation so trimesh scene.save_image() can run successfully"
        )
        return None

    png = io.BytesIO(png)
    image = Image.open(png)
    if wandb_format:
        return wandb.Image(image)
    else:
        return image


# output export_blob in .obj file type for the input trimesh voxelmesh
def mesh2wandb3D(voxelmesh, wandb_format=True):
    voxelmeshfile = voxelmesh.export(file_type="obj")
    if not wandb_format:
        return voxelmeshfile
    else:
        voxelmeshfile = wandb.Object3D(io.StringIO(voxelmeshfile), file_type="obj")
        return voxelmeshfile


# helper function for wandbLog
# calculate mesh data statistics
# 1) average number of voxel per tree
# 2) average number of voxel cluster per tree (check distance function)
# 3) images of all generated tree
# 4) meshes of all generated tree
# 5) mean of per voxel std over generated trees
def calculate_log_media_stat(samples):
    num_samples = samples.shape[0]
    # log_dict list record
    sample_tree_numpoints = []
    eval_num_cluster = []
    sample_tree_image = []
    sample_tree_voxelmesh = []
    for n in range(num_samples):
        # samples are before sigmoid
        sample_tree_bool_array = samples[n] > 0
        # log number of points to wandb
        sample_tree_indices = netarray2indices(sample_tree_bool_array)
        sample_tree_numpoints.append(sample_tree_indices.shape[0])
        # count number of cluster in the tree (grouped with dist_inf = 1)
        num_cluster = eval_count_cluster(sample_tree_bool_array)
        eval_num_cluster.append(num_cluster)

        voxelmesh = netarray2mesh(sample_tree_bool_array)

        # image / 3D object to log_dict
        image = mesh2wandbImage(voxelmesh)
        if image is not None:
            sample_tree_image.append(image)
        voxelmeshfile = mesh2wandb3D(voxelmesh)
        sample_tree_voxelmesh.append(voxelmeshfile)

    # mean
    sample_tree_numpoints = np.mean(sample_tree_numpoints)
    eval_num_cluster = np.mean(eval_num_cluster)
    # mesh model variance
    mesh_bool_array = samples > 0
    mesh_per_voxel_std = np.mean(np.std(mesh_bool_array, 0))

    return (
        sample_tree_numpoints,
        eval_num_cluster,
        sample_tree_image,
        sample_tree_voxelmesh,
        mesh_per_voxel_std,
    )


#####
#   helper function (datamodule)
#####

# given a mesh (trimesh), this function rotate the mesh by
# (radians[0], axes[0]), then (radians[1], axes[1]), ...
def rotateMesh(voxelmesh, radians, axes):
    assert len(radians) == len(axes)
    for i in range(len(axes)):
        ra = radians[i]
        ax = axes[i]
        voxelmesh = voxelmesh.apply_transform(
            trimesh.transformations.rotation_matrix(ra, ax)
        )
    return voxelmesh


# given a mesh (trimesh), this function voxelize the mesh
# into an array_length**3 cube in the form of boolean array
def mesh2arrayCentered(mesh, array_length, voxel_size=1):
    # given array length 64, voxel size 2, then output array size is [128,128,128]
    resolution = np.ceil(
        np.array([array_length, array_length, array_length]) / voxel_size
    ).astype(int)
    vox_array = np.zeros(
        resolution, dtype=bool
    )  # tanh: voxel representation [-1,1], sigmoid: [0,1]
    # scale mesh extent to fit array_length
    max_length = np.max(np.array(mesh.extents))
    mesh = mesh.apply_transform(
        trimesh.transformations.scale_matrix((array_length - 1.5) / max_length)
    )  # now the extent is [array_length**3]
    v = mesh.voxelized(voxel_size)  # max voxel array length = array_length / voxel_size

    # find indices in the v.matrix to center it in vox_array
    indices = ((resolution - v.matrix.shape) / 2).astype(int)
    vox_array[
        indices[0] : indices[0] + v.matrix.shape[0],
        indices[1] : indices[1] + v.matrix.shape[1],
        indices[2] : indices[2] + v.matrix.shape[2],
    ] = v.matrix

    return vox_array
