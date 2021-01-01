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
#   callbacks
#####
class SaveWandbCallback(Callback):
    def __init__(self, log_interval, save_model_path):
        super().__init__()
        self.epoch = 0
        self.log_interval = log_interval
        self.save_model_path = save_model_path

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if self.epoch % self.log_interval == 0:
            # log
            trainer.save_checkpoint(self.save_model_path)
            save_checkpoint_to_cloud(self.save_model_path)
        self.epoch += 1


#####
#   functions
#####


def load_dataset(dataset_name, run, config):
    # download dataset
    dataset_artifact = run.use_artifact(dataset_name, type="dataset")
    dir_dict = dataset_artifact.metadata["dir_dict"]
    artifact_dir = dataset_artifact.download()

    # process
    dataset = []
    for data_cat in dir_dict:
        filename_list = dir_dict[data_cat]
        for filename in filename_list:
            filename = artifact_dir + "/" + data_cat + "/" + filename
            m = trimesh.load(filename, force="mesh")
            # augment data
            if config.data_augmentation:
                array = data_augmentation(
                    m,
                    num_augment_data=config.num_augment_data,
                    array_length=config.resolution,
                )
            else:
                array = mesh2arrayCentered(m, array_length=config.resolution)[
                    np.newaxis, :, :, :
                ]
            dataset.append(array)

    # now all the returned array contains multiple samples
    dataset = np.concatenate(dataset)
    return dataset


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


# helper function for wandbLog
# calculate mesh data statistics
# 1) average number of voxel per tree
# 2) average number of voxel cluster per tree (check distance function)
# 3) images of all generated tree
# 4) meshes of all generated tree
# 5) mean of per voxel std over generated trees
def calculate_log_media_stat(model, log_num_samples, class_label=None):
    if class_label is not None:
        sample_trees = model.generate_tree(c=class_label, num_trees=log_num_samples)
    else:
        sample_trees = model.generate_tree(num_trees=log_num_samples)

    # log_dict list record
    sample_tree_numpoints = []
    eval_num_cluster = []
    sample_tree_image = []
    sample_tree_voxelmesh = []
    for n in range(log_num_samples):
        # sample_trees are before sigmoid
        sample_tree_bool_array = sample_trees[n] > 0
        # log number of points to wandb
        sample_tree_indices = netarray2indices(sample_tree_bool_array)
        sample_tree_numpoints.append(sample_tree_indices.shape[0])
        # mean
        sample_tree_numpoints = np.mean(sample_tree_numpoints)
        # count number of cluster in the tree (grouped with dist_inf = 1)
        num_cluster = eval_count_cluster(sample_tree_bool_array)
        eval_num_cluster.append(num_cluster)
        # mean
        eval_num_cluster = np.mean(eval_num_cluster)

        voxelmesh = netarray2mesh(sample_tree_bool_array)

        # image / 3D object to log_dict
        image = mesh2wandbImage(voxelmesh)
        if image is not None:
            sample_tree_image.append(image)
        voxelmeshfile = mesh2wandb3D(voxelmesh)
        sample_tree_voxelmesh.append(voxelmeshfile)

        # mesh model variance
        mesh_bool_array = sample_trees > 0
        mesh_per_voxel_std = np.mean(np.std(mesh_bool_array, 0))
    return (
        sample_tree_numpoints,
        eval_num_cluster,
        sample_tree_image,
        sample_tree_voxelmesh,
        mesh_per_voxel_std,
    )


def save_checkpoint_to_cloud(checkpoint_path):
    wandb.save(checkpoint_path)


def load_checkpoint_from_cloud(checkpoint_path="model_dict.pth"):
    checkpoint_file = wandb.restore(checkpoint_path)
    return checkpoint_file.name


#####
#   helper function (array processing and log)
#####
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


# array should be 3d
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


def mesh2wandb3D(voxelmesh, wandb_format=True):
    voxelmeshfile = voxelmesh.export(file_type="obj")
    if not wandb_format:
        return voxelmeshfile
    else:
        voxelmeshfile = wandb.Object3D(io.StringIO(voxelmeshfile), file_type="obj")
        return voxelmeshfile


#####
#   helper function (dataset)
#####
def rotateMesh(voxelmesh, radians, axes):
    assert len(radians) == len(axes)
    for i in range(len(axes)):
        ra = radians[i]
        ax = axes[i]
        voxelmesh = voxelmesh.apply_transform(
            trimesh.transformations.rotation_matrix(ra, ax)
        )
    return voxelmesh


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


def data_augmentation(mesh, array_length, num_augment_data=4, scale_max_margin=3):

    retval = np.zeros((num_augment_data, array_length, array_length, array_length))

    for i in range(num_augment_data):

        # first select rotation angle (angle in radian)
        angle = 2 * np.pi * (np.random.rand(1)[0])

        # scale is implemented based on the bounding box with box margin (larger margin, smaller scale)
        box_margin = np.random.randint(scale_max_margin + 1)

        # pick a random starting point within margin as translation
        initial_position = np.random.randint(box_margin + 1, size=3)

        result_array = modify_mesh(
            mesh, array_length, angle, box_margin, initial_position
        )
        retval[i] = result_array

    return retval


def modify_mesh(mesh, out_array_length, rot_angle, scale_box_margin, array_init_pos):
    # first copy mesh
    mesh = mesh.copy()
    # rotate mesh by rot_angle in radian
    mesh = mesh.apply_transform(
        trimesh.transformations.rotation_matrix(rot_angle, (0, 1, 0))
    )

    # scale is implemented based on the bounding box with box margin (larger margin, smaller scale)
    # example (assume out_array_length=64): margin = 0, bounding box shape = (64,64,64); margin = 3, bounding box shape = (61,61,61)
    scaled_size = out_array_length - scale_box_margin
    mesh_array = mesh2arrayCentered(mesh, array_length=scaled_size)

    # put them into bounding box (and translation)
    retval = np.zeros((out_array_length, out_array_length, out_array_length))
    # apply translation by selecting initial position
    # example: same mesh array of size (61,61,61) but with two position (0,1,0) and (1,0,0) is just a translation of 2 units
    x, y, z = array_init_pos
    retval[x : x + scaled_size, y : y + scaled_size, z : z + scaled_size] = mesh_array

    return retval
