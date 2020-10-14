import os
import io
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
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks.base import Callback

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


#####
#   DataModule
#####
class DataModule_process(pl.LightningDataModule):
    def __init__(self, config, run, filepath):
        super().__init__()
        self.config = config
        self.run = run
        self.dataset_artifact = None
        self.dataset = None
        self.size = 0
        self.filepath = filepath

    def prepare_data(self):
        file_ext = [
            ".obj",
            ".off",
            ".ply",
            ".stl",
            ".dae",
            ".misc",
            ".gltf",
            ".assimp",
            ".threemf",
            ".openctm",
            ".xml_based",
            ".binvox",
            ".xyz",
        ]

        # array to hold process information
        data = []
        failed = []
        dataset_array = []

        if self.filepath.endswith(".zip"):

            # process zipfile path information
            zipfile_loc_list = self.filepath.split("/")
            zipfile_name = zipfile_loc_list[-1]
            zipfile_title = ".".join(zipfile_name.split(".")[:-1])
            # check if exist .npy (the npy and zip files should be in the same location)
            npy_path = os.path.join(
                "/".join(zipfile_loc_list[:-1]), zipfile_title + ".npy"
            )
            if os.path.isfile(npy_path):
                print(zipfile_title + ".npy file already exists!")
                self.filepath = npy_path
                return
            # process zip file
            zf = zipfile.ZipFile(self.filepath, "r")
            for file_name in zf.namelist():
                for ext in file_ext:
                    if file_name.endswith(ext):
                        try:
                            # print(file_name)
                            file = zf.open(file_name, "r")
                            file = BytesIO(file.read())
                            m = trimesh.load(file, file_type=ext[1:], force="mesh")
                            array = mesh2arrayCentered(m, array_length=32)
                            # #get filename that can be read by trimesh
                            data.append(file_name)
                            dataset_array.append(array)
                        except IndexError:
                            failed.append(file_name)
                            print(file_name + " failed")
                        # the file is processed with corresponding extension
                        break

        else:
            # this is a file folder
            # process zipfile path information
            if self.filepath[-1] == "/":
                self.filepath = self.filepath[:-1]
            npy_path = os.path.join(self.filepath, "dataset_array_processed.npy")
            if os.path.isfile(npy_path):
                print("dataset_array_processed.npy file already exists!")
                self.filepath = npy_path
                return
            # process file in self.filepath
            for file_name in os.listdir(self.filepath):
                for ext in file_ext:
                    if file_name.endswith(ext):
                        try:
                            m = trimesh.load(
                                os.path.join(self.filepath, file_name), force="mesh"
                            )
                            array = mesh2arrayCentered(m, array_length=32)
                            # print(array.shape)
                            # get filename that can be read by trimesh
                            data.append(file_name)
                            dataset_array.append(array)
                        except IndexError:
                            failed.append(file_name)
                            print(file_name + " failed")
                        # the file is processed with corresponding extension
                        break

        # save as numpy array
        dataset_array = np.stack(dataset_array, axis=0)
        np.save(npy_path, dataset_array)
        self.filepath = npy_path
        print("processed dataset_array shape: " + str(dataset_array.shape))
        print("number of failed data: " + str(len(failed)))
        return

    def setup(self, stage=None):
        config = self.config
        dataset = np.load(self.filepath)

        # now all the returned array contains multiple samples
        self.size = dataset.shape[0]
        self.dataset = torch.unsqueeze(torch.tensor(dataset), 1)

    def train_dataloader(self):
        config = self.config
        tensor_dataset = TensorDataset(self.dataset)
        return DataLoader(tensor_dataset, batch_size=config.batch_size, shuffle=True)


class DataModule_custom_cond(pl.LightningDataModule):
    def __init__(self, config, run, data_filename, index_filename):
        super().__init__()
        self.config = config
        self.run = run
        self.dataset_artifact = None
        self.dataset = None
        self.data_index = None
        self.size = 0
        self.num_classes = 0
        self.data_filename = data_filename
        self.index_filename = index_filename

    def prepare_data(self):
        return

    def setup(self, stage=None):
        config = self.config
        dataset = np.load(self.data_filename)
        data_index_dict = np.load(self.index_filename, allow_pickle=True)

        # now all the returned array contains multiple samples
        self.size = dataset.shape[0]
        self.dataset = torch.unsqueeze(torch.tensor(dataset), 1)

        # process data index dict in the form {"double_trunk":[1,5,6], ....} to class index array [0, 1, 3, 1, 2, 0]
        data_index = np.zeros(self.size)
        data_index_dict = data_index_dict.item()
        for class_num, key in enumerate(data_index_dict.keys()):
            class_list = data_index_dict[key]
            for i in class_list:
                data_index[i] = class_num

        config.num_classes = class_num + 1
        self.data_index = torch.tensor(data_index).int()

    def train_dataloader(self):
        config = self.config
        tensor_dataset = TensorDataset(self.dataset, self.data_index)
        return DataLoader(tensor_dataset, batch_size=config.batch_size, shuffle=True)


class DataModule_custom(pl.LightningDataModule):
    def __init__(self, config, run, filename):
        super().__init__()
        self.config = config
        self.run = run
        self.dataset_artifact = None
        self.dataset = None
        self.size = 0
        self.filename = filename

    def prepare_data(self):
        return

    def setup(self, stage=None):
        config = self.config
        dataset = np.load(self.filename)

        # now all the returned array contains multiple samples
        self.size = dataset.shape[0]
        self.dataset = torch.unsqueeze(torch.tensor(dataset), 1)

    def train_dataloader(self):
        config = self.config
        tensor_dataset = TensorDataset(self.dataset)
        return DataLoader(tensor_dataset, batch_size=config.batch_size, shuffle=True)


class DataModule(pl.LightningDataModule):
    def __init__(self, config, run):
        super().__init__()
        self.config = config
        self.run = run
        self.dataset_artifact = None
        self.dataset = None
        self.size = 0

    def prepare_data(self):
        # download
        run = self.run
        dataset_name = self.config.dataset
        self.dataset_artifact = run.use_artifact(dataset_name, type="dataset")

    def setup(self, stage=None):
        config = self.config
        dir_dict = self.dataset_artifact.metadata["dir_dict"]
        artifact_dir = self.dataset_artifact.download()

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
                        array_length=config.array_size,
                    )
                else:
                    array = mesh2arrayCentered(m, array_length=config.array_size)[
                        np.newaxis, :, :, :
                    ]
                dataset.append(array)

        # now all the returned array contains multiple samples
        dataset = np.concatenate(dataset)
        self.size = dataset.shape[0]
        self.dataset = torch.unsqueeze(torch.tensor(dataset), 1)

    def train_dataloader(self):
        config = self.config
        tensor_dataset = TensorDataset(self.dataset)
        return DataLoader(tensor_dataset, batch_size=config.batch_size, shuffle=True)


#####
#   callbacks
#####
class SaveWandbCallback(Callback):
    def __init__(self, log_interval, save_model_path):
        super().__init__()
        self.epoch = 0
        self.log_interval = log_interval
        self.save_model_path = save_model_path

    def on_train_epoch_end(self, trainer, pl_module):
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
                    array_length=config.array_size,
                )
            else:
                array = mesh2arrayCentered(m, array_length=config.array_size)[
                    np.newaxis, :, :, :
                ]
            dataset.append(array)

    # now all the returned array contains multiple samples
    dataset = np.concatenate(dataset)
    return dataset


def wandbLog(model, initial_log_dict={}, log_image=False, log_mesh=False):

    if log_image or log_mesh:
        sample_tree_array = model.generate_tree()[0]  # only 1 tree
        sample_tree_indices = netarray2indices(sample_tree_array)
        # log number of points to wandb
        initial_log_dict["sample_tree_numpoints"] = sample_tree_indices.shape[0]
        voxelmesh = netarray2mesh(sample_tree_array)

        if log_image:
            image = mesh2wandbImage(voxelmesh)
            initial_log_dict["sample_tree_image"] = image

        if log_mesh:
            voxelmeshfile = mesh2wandb3D(voxelmesh)
            initial_log_dict["sample_tree_voxelmesh"] = voxelmeshfile

    wandb.log(initial_log_dict)


def wandbLog_cond(model, c, initial_log_dict={}, log_image=False, log_mesh=False):

    if log_image or log_mesh:
        sample_tree_array = model.generate_tree(c)[0]  # only 1 tree
        sample_tree_indices = netarray2indices(sample_tree_array)
        # log number of points to wandb
        initial_log_dict["sample_tree_class"] = c.detach().cpu().numpy()
        initial_log_dict["sample_tree_numpoints"] = sample_tree_indices.shape[0]
        voxelmesh = netarray2mesh(sample_tree_array)

        if log_image:
            image = mesh2wandbImage(voxelmesh)
            initial_log_dict["sample_tree_image"] = image

        if log_mesh:
            voxelmeshfile = mesh2wandb3D(voxelmesh)
            initial_log_dict["sample_tree_voxelmesh"] = voxelmeshfile

    wandb.log(initial_log_dict)


def save_checkpoint_to_cloud(checkpoint_path):
    wandb.save(checkpoint_path)


def load_checkpoint_from_cloud(checkpoint_path="model_dict.pth"):
    checkpoint_file = wandb.restore(checkpoint_path)
    return checkpoint_file.name


#####
#   helper function (array processing and log)
#####
def netarray2indices(array):
    coord_list = []
    if len(array.shape) == 5:
        array = array[0][0]
    x, y, z = array.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                if (
                    array[i, j, k] > 0.5
                ):  # tanh: voxel representation [-1,1], sigmoid: [0,1]
                    coord_list.append([i, j, k])
    # print(len(coord_list))
    if len(coord_list) == 0:
        return np.array(
            [[0, 0, 0]]
        )  # return at least one point to prevent wandb 3dobject error
    return np.array(coord_list)


# array should be 3d
def netarray2mesh(array):
    if len(array.shape) != 3:
        raise Exception("netarray2mesh: input array should be 3d")

    # convert to bool dtype
    array = array > 0.5
    # array all zero gives error
    if np.sum(array) == 0:
        array[0, 0, 0] = True
    voxelmesh = trimesh.voxel.base.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(array)
    ).marching_cubes
    return voxelmesh


def mesh2wandbImage(voxelmesh):
    scene = voxelmesh.scene()
    try:
        png = scene.save_image(
            resolution=[600, 600],
        )
    except NoSuchDisplayException:
        print(
            "NoSuchDisplayException. Renderer not found! Please check configuation so trimesh scene.save_image() can run successfully"
        )
    png = io.BytesIO(png)
    image = Image.open(png)
    return wandb.Image(image)


def mesh2wandb3D(voxelmesh):
    voxelmeshfile = voxelmesh.export(file_type="obj")
    voxelmeshfile = wandb.Object3D(io.StringIO(voxelmeshfile), file_type="obj")
    return voxelmeshfile


#####
#   helper function (dataset)
#####
def mesh2arrayCentered(mesh, voxel_size=1, array_length=64):
    # given array length 64, voxel size 2, then output array size is [128,128,128]
    array_size = np.ceil(
        np.array([array_length, array_length, array_length]) / voxel_size
    ).astype(int)
    vox_array = np.zeros(
        array_size, dtype=bool
    )  # tanh: voxel representation [-1,1], sigmoid: [0,1]
    # scale mesh extent to fit array_length
    max_length = np.max(np.array(mesh.extents))
    mesh = mesh.apply_transform(
        trimesh.transformations.scale_matrix((array_length - 1.5) / max_length)
    )  # now the extent is [array_length**3]
    v = mesh.voxelized(voxel_size)  # max voxel array length = array_length / voxel_size

    # find indices in the v.matrix to center it in vox_array
    indices = ((array_size - v.matrix.shape) / 2).astype(int)
    vox_array[
        indices[0] : indices[0] + v.matrix.shape[0],
        indices[1] : indices[1] + v.matrix.shape[1],
        indices[2] : indices[2] + v.matrix.shape[2],
    ] = v.matrix

    return vox_array


def data_augmentation(mesh, array_length=64, num_augment_data=4, scale_max_margin=3):

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
