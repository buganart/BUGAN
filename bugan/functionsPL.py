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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

#################
#       datamodule that modify data on-the-fly
#       data are processed in __getitem__() func defined in AugmentationDataset in the module
#       init: dataModule = DataModule_augmentation(config, run, data_path)
#              where data_path is the directory of all the files (not recursive)
#              example: data_path = "../../../../../My Drive/Hand-Tool-Data-Set/turbosquid_thingiverse_dataset/dataset_ply/"
#################
class DataModule_augmentation(pl.LightningDataModule):
    class AugmentationDataset(Dataset):
        def __init__(self, config, data_list):
            assert isinstance(data_list, list)
            self.data_list = data_list
            self.config = config

        def __getitem__(self, index):
            selectedItem = self.data_list[index]
            radian = 2 * np.pi * (np.random.rand(1)[0])
            # not going to copy the mesh before rotation (performance consideration)
            selectedItem = rotateMesh(
                selectedItem, [radian], [self.config.aug_rotation_axis]
            )
            array = mesh2arrayCentered(
                selectedItem, array_length=self.config.array_size
            )  # assume selectedItem is Trimesh object
            # print("mesh index:", index, "| rot radian:", angle)
            return torch.tensor(array[np.newaxis, np.newaxis, :, :, :])

        def __len__(self):
            return len(self.data_list)

        def __add__(self, other):
            return AugmentationDataset(self.config, self.data_list.append(other))

    def __init__(self, config, run, data_path):
        super().__init__()
        self.config = config
        self.run = run
        self.dataset_artifact = None
        self.dataset = None
        self.size = 0
        self.data_path = (
            Path(data_path) if data_path[-1] != "/" else Path(data_path[:-1])
        )
        self.file_ext = [
            ".ply",
            ".stl",
            ".dae",
            ".obj",
            ".off",
            ".misc",
            ".gltf",
            ".assimp",
            ".threemf",
            ".openctm",
            ".xml_based",
            ".binvox",
            ".xyz",
        ]

    def _unzip_zip_file(self):

        failed = []
        samples = []

        zf = zipfile.ZipFile(self.data_path, "r")

        # unzip all the files into a directory
        dir_path = self.data_path.parent / self.data_path.stem
        # create folder if dir not exists
        if not dir_path.exists():
            try:
                os.mkdir(dir_path)
            except OSError:
                print(f"create directory {dir_path} failed")
        zf.extractall(path=dir_path)
        # construct datapath file
        return dir_path

    def prepare_data(self):
        if self.data_path.is_dir():
            return
        elif self.data_path.suffix == ".zip":
            self.data_path = self._unzip_zip_file()
            return

    def setup(self, stage=None):
        config = self.config
        dataset = []

        paths = [
            path
            for path in self.data_path.rglob("*.*")
            if path.suffix in self.file_ext and not "__MACOSX" in str(path)
        ]

        for file_name in paths:
            try:
                m = trimesh.load(file_name, force="mesh")
                dataset.append(m)
            except Exception as e:  # TODO: check if we should report error
                print(e)
                print(str(file_name) + " failed")

        # now all the returned array contains multiple samples
        self.size = len(dataset)
        self.dataset = dataset

    def train_dataloader(self):
        config = self.config
        aug_dataset = self.AugmentationDataset(self.config, self.dataset)
        return DataLoader(aug_dataset, batch_size=config.batch_size, shuffle=True)


def make_npy_path(path: Path, res):
    if path.is_dir():
        # TODO
        # Preferably we would not save into the dataset directory it can break
        # code that relies on there not being extra files in the dataset directory.
        #
        # We could use
        #
        #     path.parent / "{path.name}.npy"
        #
        # instead or save to an entirely different location.
        return path / "dataset_array_processed_res" + str(res) + ".npy"
    elif path.suffix == ".zip":
        return path.parent / f"{path.stem}_res{res}.npy"
    elif path.suffix == ".npy":
        return path
    else:
        raise ValueError(f"Cannot handle dataset path {path}")


#####
#   DataModule
#####
class DataModule_process(pl.LightningDataModule):
    supported_extensions = set(
        [
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
    )

    def __init__(self, config, run, filepath):
        super().__init__()
        self.config = config
        self.run = run
        self.dataset_artifact = None
        self.dataset = None
        self.size = 0
        self.filepath = Path(filepath) if filepath[-1] != "/" else Path(filepath[:-1])
        self.npy_path = make_npy_path(self.filepath, self.config.array_size)

    def _read_meshes_from_zip_file(self):

        failed = []
        samples = []

        zf = zipfile.ZipFile(self.filepath, "r")
        supported_files = [
            path
            for path in zf.namelist()
            if (
                Path(path).suffix in self.supported_extensions
                and not path.startswith("__MACOSX")
            )
        ]

        for path in tqdm.tqdm(supported_files, desc="Meshes"):
            try:
                file = zf.open(path, "r")
                file = BytesIO(file.read())
                m = trimesh.load(
                    file,
                    file_type=Path(path).suffix[1:],
                    force="mesh",
                )
                array = mesh2arrayCentered(m, array_length=self.config.array_size)
                samples.append(array)
            except IndexError:
                failed.append(path)
                print("Failed to load {path}")
        return samples, failed

    def _read_meshes_from_directory(self):

        failed = []
        samples = []

        paths = [
            path
            for path in self.filepath.rglob("*.*")
            if path.suffix in self.supported_extensions
        ]

        for path in tqdm.tqdm(paths, desc="Meshes"):
            try:
                m = trimesh.load(path, force="mesh")
                array = mesh2arrayCentered(m, array_length=self.config.array_size)
                samples.append(array)
            except Exception as exc:
                failed.append(path)
                print(f"Failed to load {path}: {exc}")
        return samples, failed

    def prepare_data(self):

        if self.npy_path.exists():
            print(f"Processed dataset {self.npy_path} already exists.")
            return

        if self.filepath.suffix == ".zip":
            loader = self._read_meshes_from_zip_file
        else:
            loader = self._read_meshes_from_directory

        samples, failed = loader()
        dataset = np.array(samples)
        print(f"Processed dataset_array shape: {dataset.shape}")
        print(f"Number of failed file: {len(failed)}")

        np.save(self.npy_path, dataset)
        print(f"Saved processed dataset to {self.npy_path}")

    def setup(self, stage=None):
        dataset = np.load(self.npy_path)

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


def eval_count_cluster(array):
    def nearby_voxels(array, i, j, k):
        bound = array.shape
        low_i, high_i = np.clip([i - 1, i + 1], 0, bound[0] - 1)
        low_j, high_j = np.clip([j - 1, j + 1], 0, bound[1] - 1)
        low_k, high_k = np.clip([k - 1, k + 1], 0, bound[2] - 1)

        retval = []
        for x in range(low_i, high_i + 1):
            for y in range(low_j, high_j + 1):
                for z in range(low_k, high_k + 1):
                    if array[x, y, z]:
                        # voxel exists
                        retval.append((x, y, z))
        # remove the selected vox itself
        retval.remove((i, j, k))
        return retval

    ds = DisjointSet()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                vox = array[i, j, k]
                if vox:
                    # voxel exists in coord (i,j,k)
                    nearby_vox = nearby_voxels(array, i, j, k)
                    for v in nearby_vox:
                        ds.union(v, (i, j, k))
    return len(list(ds.itersets()))


def wandbLog(model, initial_log_dict={}, log_media=False, log_num_samples=1):

    if log_media:
        sample_trees = model.generate_tree(num_trees=log_num_samples)

        # log_dict list record
        sample_tree_numpoints = []
        eval_num_cluster = []
        sample_tree_image = []
        sample_tree_voxelmesh = []
        for n in range(log_num_samples):
            sample_tree_array = sample_trees[n]
            # log number of points to wandb
            sample_tree_indices = netarray2indices(sample_tree_array)
            sample_tree_numpoints.append(sample_tree_indices.shape[0])
            # count number of cluster in the tree (grouped with dist_inf = 1)
            num_cluster = eval_count_cluster(sample_tree_array)
            eval_num_cluster.append(num_cluster)

            voxelmesh = netarray2mesh(sample_tree_array)

            # image / 3D object to log_dict
            image = mesh2wandbImage(voxelmesh)
            sample_tree_image.append(image)
            voxelmeshfile = mesh2wandb3D(voxelmesh)
            sample_tree_voxelmesh.append(voxelmeshfile)

        # add list record to log_dict
        initial_log_dict["sample_tree_numpoints"] = sample_tree_numpoints
        initial_log_dict["eval_num_cluster"] = eval_num_cluster
        initial_log_dict["sample_tree_image"] = sample_tree_image
        initial_log_dict["sample_tree_voxelmesh"] = sample_tree_voxelmesh

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
