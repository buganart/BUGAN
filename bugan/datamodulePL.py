from bugan.functionsPL import rotateMesh, mesh2arrayCentered

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


#################
#       datamodule that modify data on-the-fly
#       data are processed in __getitem__() func defined in AugmentationDataset in the module
#       init: dataModule = DataModule_augmentation(config, run, data_path)
#              where data_path is the directory of all the files (not recursive)
#              example: data_path = "../../../../../My Drive/Hand-Tool-Data-Set/turbosquid_thingiverse_dataset/dataset_ply/"
#################
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

    def __init__(self, config, run, data_path, tmp_folder="/tmp/"):
        super().__init__()
        self.config = config
        self.run = run
        self.dataset_artifact = None
        self.dataset = None
        self.size = 0
        self.data_path = Path(data_path)
        is_zip = self.data_path.suffix == ".zip"
        self.zip_path = self.data_path if is_zip else None
        self.folder_path = Path(tmp_folder) if is_zip else self.data_path

        # process self.num_classes
        # num_classes > 0 => conditional data (int)
        # else => unconditional data (None)
        self.num_classes = None
        if hasattr(config, "num_classes"):
            if config.num_classes > 0:
                self.num_classes = config.num_classes

        self.class_list = None
        self.savefile_path = make_processed_savefile_path(
            self.data_path, self.config.resolution, max_num_classes=self.num_classes
        )

    def _unzip_zip_file_to_directory(self):
        print(f"Unzipping {self.zip_path} to {self.folder_path}")

        failed = []
        samples = []

        zf = zipfile.ZipFile(self.zip_path, "r")
        zf.extractall(path=self.folder_path)
        zf.close()

    def _read_mesh_array(self, isZip=True, process_to_array=True):

        failed = 0
        samples = []
        class_list = []
        sample_class_index = []

        zf = None
        filepath_list = None

        if isZip:
            zf = zipfile.ZipFile(self.zip_path, "r")
            filepath_list = zf.namelist()
        else:
            filepath_list = self.folder_path.rglob("*.*")

        # check files in filepath_list is supported (by extensions)
        supported_files = [
            path
            for path in filepath_list
            if (
                Path(path).suffix in self.supported_extensions
                and not str(path).startswith("__MACOSX")
            )
        ]

        # process files in the filepath_list
        for path in tqdm.tqdm(supported_files, desc="Meshes"):
            # extract label if conditional data
            if self.num_classes is not None:
                label = Path(path).parent.stem
                if label in class_list:
                    index = class_list.index(label)
                else:
                    class_list.append(label)
                    index = class_list.index(label)

            # process mesh
            try:
                if isZip:
                    file = zf.open(path, "r")
                    file = BytesIO(file.read())
                    m = trimesh.load(
                        file,
                        file_type=Path(path).suffix[1:],
                        force="mesh",
                    )
                else:
                    m = trimesh.load(path, force="mesh")

                # process_to_array make m to be in numpy tensor (shape: resolution**3)
                # if not process_to_array, m is in trimesh "trimesh" type
                if process_to_array:
                    m = mesh2arrayCentered(m, array_length=self.config.resolution)
                samples.append(m)
                # also append index for conditional data
                if self.num_classes is not None:
                    sample_class_index.append(index)
            except IndexError:
                failed += 1
                print("Failed to load {path}")

        if self.num_classes is not None:
            return samples, sample_class_index, failed, class_list
        else:
            return samples, failed

    def _trim_dataset(self, samples, sample_class_index, class_name_list):
        # find class_index counts
        indices, indices_count = np.unique(sample_class_index, return_counts=True)
        # sort class_index with counts
        count_list = [(indices[i], indices_count[i]) for i in range(len(indices))]
        count_list.sort(key=lambda v: v[1], reverse=True)
        # trim class_index list
        selected_class_list = count_list[: self.num_classes]
        selected_class_list = [index for (index, _) in selected_class_list]
        # shift class_name according to the selected_class_list
        class_name_list = [class_name_list[index] for index in selected_class_list]

        # trim dataset
        data = []
        index = []
        for i in range(len(sample_class_index)):
            ind = sample_class_index[i]
            if ind in selected_class_list:
                data.append(samples[i])
                # find the position of the index in selected_class_list
                # Note that the index is based on the original processed dataset,
                # not the trimmed one. So we use position of selected_class_list to
                # make sure index not out of bound
                pos = selected_class_list.index(ind)
                index.append(pos)
        return data, index, class_name_list

    # prepare_data() should contains code that will be run once per dataset.
    # most of the code will be skipped in subsequent run.
    def prepare_data(self):

        if self.config.data_augmentation:
            # for data_augmentation:
            # put/unzip all 3D objects to a directory. Ready for setup() to read
            # after perpare_data(), the target should be a directory with all 3D object files
            if self.zip_path:
                self._unzip_zip_file_to_directory()

    # setup() should contains code that will be run once per run.
    def setup(self, stage=None):

        if self.config.data_augmentation:

            if self.num_classes is None:
                # read uncondtional data
                dataset, failed = self._read_mesh_array(
                    isZip=False, process_to_array=False
                )

                # now all the returned array contains multiple samples
                self.size = len(dataset)
                self.dataset = dataset
                print(f"Processed dataset size: {len(dataset)}")
                print(f"Number of failed file: {failed}")
            else:
                # read conditional data
                (
                    dataset,
                    sample_class_index,
                    failed,
                    class_list,
                ) = self._read_mesh_array(isZip=False, process_to_array=False)

                # now all the returned array contains multiple samples
                print(f"Processed dataset size: {len(dataset)}")
                print(f"Processed number of classes: {len(set(sample_class_index))}")
                print(f"Number of failed file: {failed}")

                if self.num_classes > len(set(sample_class_index)):
                    raise ValueError(
                        f"max_num_classes ({self.num_classes}) should be <= Processed number of classes ({len(set(sample_class_index))})"
                    )
                print(
                    f"select {self.num_classes} out of {len(set(sample_class_index))} classes:"
                )

                data, index, class_list = self._trim_dataset(
                    dataset, sample_class_index, class_list
                )
                self.size = len(data)
                self.dataset = data
                self.datalabel = index

                print(f"Final dataset_array shape: {len(data)}")
                print(f"Final number of classes: {self.num_classes}")
                print("class_list:", class_list)
                self.class_list = class_list

        else:
            # for normal:
            # read all files and process the object array to .npy/.npz file
            if self.savefile_path.exists():
                print(f"Processed dataset {self.savefile_path} already exists.")
            else:

                # check input path is a zip file or a directory
                if self.data_path.suffix == ".zip":
                    isZip = True
                else:
                    isZip = False

                # read data
                if self.num_classes is not None:
                    # conditional data
                    (
                        samples,
                        sample_class_index,
                        failed,
                        class_list,
                    ) = self._read_mesh_array(isZip=isZip, process_to_array=True)
                else:
                    samples, failed = self._read_mesh_array(
                        isZip=isZip, process_to_array=True
                    )

                # print processed data information
                print(f"Processed dataset_array shape: {len(samples)}")
                print(f"Number of failed file: {failed}")

                if self.num_classes is None:
                    # just save unconditional data into .npy file
                    dataset = np.array(samples)
                    np.save(self.savefile_path, dataset)
                else:
                    # print also class information if conditional
                    print(
                        f"Processed number of classes: {len(set(sample_class_index))}"
                    )

                    if self.num_classes > len(set(sample_class_index)):
                        raise ValueError(
                            f"max_num_classes ({self.num_classes}) should be <= Processed number of classes ({len(set(sample_class_index))})"
                        )
                    print(
                        f"select {self.num_classes} out of {len(set(sample_class_index))} classes:"
                    )
                    # trim dataset by class (c = max_num_classes)
                    # only keep c classes that has highest number of samples
                    data, index, class_list = self._trim_dataset(
                        samples, sample_class_index, class_list
                    )

                    print(f"Final dataset_array shape: {len(data)}")
                    print(f"Final number of classes: {self.num_classes}")
                    print("class_list:", class_list)
                    # save as .npz file for conditional data
                    np.savez(
                        self.savefile_path,
                        data=data,
                        index=index,
                        class_list=class_list,
                    )

                print(f"Saved processed dataset to {self.savefile_path}")

            if self.num_classes is None:
                dataset = np.load(self.savefile_path)

                # now all the returned array contains multiple samples
                self.size = dataset.shape[0]
                self.dataset = torch.unsqueeze(torch.tensor(dataset), 1)
            else:
                dataFile = np.load(self.savefile_path)
                data = dataFile["data"]
                index = dataFile["index"]
                class_list = dataFile["class_list"]

                # now all the returned array contains multiple samples
                self.size = data.shape[0]
                self.dataset = torch.unsqueeze(torch.tensor(data), 1)
                self.datalabel = torch.tensor(index)
                self.class_list = class_list

    def train_dataloader(self):
        if self.config.data_augmentation:
            config = self.config

            # for conditional data, also load datalabel
            if self.num_classes is None:
                aug_dataset = AugmentationDataset(self.config, self.dataset)
            else:
                aug_dataset = AugmentationDataset(
                    self.config, self.dataset, self.datalabel
                )

            return DataLoader(
                aug_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8
            )
        else:
            config = self.config
            # for conditional data, also load datalabel
            if self.num_classes is None:
                tensor_dataset = TensorDataset(self.dataset)
            else:
                tensor_dataset = TensorDataset(self.dataset, self.datalabel)

            return DataLoader(
                tensor_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=8,
            )


#####
#   helper class
#####


class AugmentationDataset(Dataset):
    def __init__(self, config, data_list, datalabel=None):
        assert isinstance(data_list, list)
        self.data_list = data_list
        self.config = config
        self.rotation_type = config.aug_rotation_type
        self.datalabel = datalabel

        if self.rotation_type not in ["random rotation", "axis rotation"]:
            raise ValueError(
                f"aug_rotation_type should be one of ['random rotation', 'axis rotation'], current {self.rotation_type}"
            )

    def __getitem__(self, index):
        selectedItem = self.data_list[index]
        # not going to copy the mesh before rotation (performance consideration)
        if self.rotation_type == "axis rotation":
            # axis rotation
            radian = 2 * np.pi * (np.random.rand(1)[0])
            selectedItem = rotateMesh(
                selectedItem, [radian], [self.config.aug_rotation_axis]
            )
        else:
            # random rotation
            radian = 2 * np.pi * (np.random.rand(3))
            selectedItem = rotateMesh(
                selectedItem,
                [radian[0], radian[1], radian[2]],
                [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            )

        array = mesh2arrayCentered(
            selectedItem, array_length=self.config.resolution
        )  # assume selectedItem is Trimesh object
        # print("mesh index:", index, "| rot radian:", angle)
        if self.datalabel:
            return torch.tensor(array[np.newaxis, :, :, :]), torch.tensor(
                self.datalabel[index]
            )
        else:
            return torch.tensor(array[np.newaxis, np.newaxis, :, :, :])

    def __len__(self):
        return len(self.data_list)

    def __add__(self, other):
        if self.datalabel:
            data, label = other
            return AugmentationDataset(
                self.config, self.data_list.append(data), self.datalabel.append(label)
            )
        else:
            return AugmentationDataset(self.config, self.data_list.append(other))


# return npy file for unconditional data
# return npz file for conditional data
def make_processed_savefile_path(path: Path, res, max_num_classes=None):
    # TODO
    # Preferably we would not save into the dataset directory it can break
    # code that relies on there not being extra files in the dataset directory.
    #
    # We could use
    #
    #     path.parent / "{path.name}.npy"
    #
    # instead or save to an entirely different location.
    if max_num_classes is None:
        # unconditional data
        if path.is_dir():
            return path / (f"dataset_array_processed_res{res}.npy")
        elif path.suffix == ".zip":
            return path.parent / f"{path.stem}_res{res}.npy"
        elif path.suffix == ".npy":
            return path
        else:
            raise ValueError(f"Cannot handle dataset path {path}")
    else:
        # conditional data
        if path.is_dir():
            return path / (f"dataset_array_processed_res{res}_c{max_num_classes}.npz")
        elif path.suffix == ".zip":
            return path.parent / (f"{path.stem}_res{res}_c{max_num_classes}.npz")
        elif path.suffix == ".npz":
            return path
        else:
            raise ValueError(f"Cannot handle dataset path {path}")
