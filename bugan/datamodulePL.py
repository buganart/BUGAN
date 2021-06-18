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


#####
#   DataModule
#####
class DataModule_process(pl.LightningDataModule):
    """
    pytorch lightning datamodule to load mesh files from zipfile/directory specified by data_path

    If data_augmentation=False, this datamodule will process the data into array and store in .npy/.npz file.
        The location of the .npy/.npz file is created by make_processed_savefile_path().
    If True, this will just load all mesh files into a list for AugmentationDataset to perform data_augmentation on-the-fly.
        data are processed in __getitem__() func defined in AugmentationDataset in the module

    This datamodule will load data as unconditional data by default (or config.num_classes <= 0).
    If config.num_classes > 0, data will be loaded as conditional data. The label of the data is the folder name containing it.
    For example: "zipFile/data/oak_tree/tree_1.obj" has "oak_tree" as label

    This datamodule will return a torch dataloader (after train_dataloader() is called).
    For unconditional, the dataset_batch of the dataloader is [array], which the array represents the processed meshes
    For conditional, the dataset_batch of the dataloader is [array, index], which the index represents the class index from the class_list
    class_list will be created during the processing of the files (calling setup())

    Attributes
    ----------
    config : Namespace
        dictionary of training parameters
    config.num_classes : int
        indicate maximum number of classes to read.
        If num_classes <= 0, the data is unconditional, and dataset_batch will not contain index
        If num_classes > 0, the data is conditional, and dataset_batch will contain index
        Assume the datamodule can read n classes from the zipfile/directory specified by the data_path,
            if n > num_classes, the _trim_dataset() will be called. The class_index will be sorted by data count.
            the (n - num_classes) classes with least data count will be removed.
            if n < num_classes, raise ValueError()
    config.batch_size : int
        the batch_size for the dataloader
    config.resolution : int
        the size of the array for voxelization
        if resolution=32, the resulting array of processing N mesh files are (N, 1, 32, 32, 32)
    config.data_augmentation : boolean
        whether to apply data_augmentation or not
        if data_augmentation = True, the array processing will be done on-the-fly (see AugmentationDataset below)
            the augmentation actually do not create new meshes, but just rotate the original mesh based on aug_rotation_type, aug_rotation_axis
        if data_augmentation = False, the mesh will be processed into array in setup().
        The array of unconditional data will be store in .npy
        The array, index, and class_list of conditional data will be stored in .npz (npzfile['data'],  npzfile['index'], npzfile['class_list'])
    config.aug_rotation_type : string
        argument for AugmentationDataset to determine rotation_type of data_augmentation
    config.aug_rotation_axis : (float, float, float)
        argument for AugmentationDataset to determine rotation_axis of data_augmentation if aug_rotation_type="axis_roataion"

    data_path : string
        the relative/absolute path specifies the location (zipfile/directory) of mesh files
        example for zipfile: "../Hand-Tool-Data-Set/turbosquid_thingiverse_dataset/dataset_ply_out.zip"
        example for directory: "../Hand-Tool-Data-Set/turbosquid_thingiverse_dataset/dataset_ply/"
            with mesh files in the folder: "....../dataset_ply/pen/pen1.obj"

    tmp_folder : string
        the folder for zipFile to temporarly extract files on if data_augmentation = True.
        If data_augmentation = True and data_path points to zip, the mesh files will be extracted to the location when prepare_data() is called
        else, tmp_folder is ignored.

    Methods
    -------
    _unzip_zip_file_to_directory()
        unzip files in the zip file
    _read_mesh_array(isZip=True, process_to_array=True)
        read and process mesh files into array
    _trim_dataset(samples, sample_class_index, class_name_list)
        trim the dataset based on self.num_classes
    prepare_data()
        default function to download data / write to disk for pytorch datamodule
    setup(stage=None)
        default function to process data and assign them as attribute
    train_dataloader():
        default function for pytorch datamodule to return torch dataloader
    """

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

    def __init__(self, config, data_path, tmp_folder="/tmp/"):
        super().__init__()
        self.config = config
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

        # trim class_index list
        if hasattr(self.config, "trim_class_offset"):
            self.offset = self.config.trim_class_offset
        else:
            self.offset = None

        # selected classes to train
        if hasattr(self.config, "selected_class_name_list"):
            self.class_list = self.config.selected_class_name_list
            # reassign self.num_classes if selected classes
            if len(self.class_list) != self.num_classes:
                raise ValueError(
                    f"the length of selected_class_name_list ({len(self.class_list)}) should be the same as num_classes ({self.num_classes})"
                )
        else:
            self.class_list = None

        self.savefile_path = make_processed_savefile_path(
            self.data_path,
            self.config.resolution,
            max_num_classes=self.num_classes,
            offset=self.offset,
        )

    def _unzip_zip_file_to_directory(self):
        """
        unzip files in the zip file

        Parameters
        ----------
        self.zip_path : string
            the path to the target zip file
        self.folder_path : string
            the path to the output directory
        """
        print(f"Unzipping {self.zip_path} to {self.folder_path}")

        failed = []
        samples = []

        zf = zipfile.ZipFile(self.zip_path, "r")
        zf.extractall(path=self.folder_path)
        zf.close()

    def _read_mesh_array(self, isZip=True, process_to_array=True):
        """
        read and process mesh files into array

        Parameters
        ----------
        isZip : boolean
            indicate whether the data_path specifies a zip file or a directory
        process_to_array: boolean
            whether to process the loaded mesh files into numpy array or not
        config.resolution : int
            the size of the array for voxelization
        config.num_classes : int
            indicate maximum number of classes to read whether to process data with label or not
        self.zip_path : Path
        self.folder_path : Path
            the location of the mesh files. If isZip=True, self.zip_path is used.

        Returns
        -------
        samples : list/numpy ndarray
            if process_to_array=True, all successfully loaded mesh files will be voxelized into numpy array
            if process_to_array=False, all successfully loaded mesh files will be appended to a list
        sample_class_index : list of int
            the class indices of the loaded mesh files above.
            sample_class_index[i] is the class index of samples[i]
            if self.num_classes is None, this will not return
        failed : int
            the number of mesh files failed to load
        class_list : list of string
            the string label of the class index.
            class_list[sample_class_index[i]] is the label of samples[i]

        """

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
                if str(label) == "_unconditional":
                    index = -1
                elif label in class_list:
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
            except:
                failed += 1
                print(f"Failed to load {path}")

        if self.num_classes is not None:
            return samples, sample_class_index, failed, class_list
        else:
            return samples, failed

    def _trim_dataset(self, samples, sample_class_index, class_name_list):
        """
        Trim the dataset based on self.num_classes
        This function will first produce the count of each class_index,
        and then sort class_index by count.
        Assume that self.num_classes < len(class_name_list),
        this function will remove classes with least count until the number of classes = self.num_classes
        Parameters
        ----------
        samples : list/numpy ndarray
            loaded mesh data processed from _read_mesh_array()
            if process_to_array=True, all successfully loaded mesh files will be voxelized into numpy array
            if process_to_array=False, all successfully loaded mesh files will be appended to a list
        sample_class_index : list of int
            the class indices of the loaded mesh files from _read_mesh_array()
            sample_class_index[i] is the class index of samples[i]
            if self.num_classes is None, this will not return
        class_name_list : list of string
            the string label of the class index from _read_mesh_array()
            class_list[sample_class_index[i]] is the label of samples[i]

        Returns
        -------
        data : list/numpy ndarray
            the sorted samples list but with trimmed classes
        index : list of int
            the sample_class_index list but with trimmed classes
        class_name_list : list of string
            the class_name_list list but with trimmed classes
        """

        # if self.class_list is assigned, then there is selected classes for the training
        if self.class_list is None:
            # find class_index counts
            indices, indices_count = np.unique(sample_class_index, return_counts=True)
            count_list = [(indices[i], indices_count[i]) for i in range(len(indices))]

            # # sort class_index with counts
            # count_list.sort(key=lambda v: v[0])
            # count_list.sort(key=lambda v: v[1], reverse=True)
            if not hasattr(self.config, "seed"):
                self.config.seed = 123
            np.random.seed(self.config.seed)
            count_list = np.random.permutation(count_list)

            if self.offset is None:
                offset = 0
            else:
                offset = self.offset

            if offset >= len(count_list):
                raise ValueError(
                    f"trim_class_offset ({offset}) should be <= Processed number of classes ({len(count_list)})"
                )
            if self.num_classes + offset >= len(count_list):
                selected_class_list = count_list[offset:]
            else:
                selected_class_list = count_list[offset : self.num_classes + offset]

            selected_class_list = [index for (index, _) in selected_class_list]
            # shift class_name according to the selected_class_list
            class_name_newlist = []
            for index in selected_class_list:
                if index >= 0:
                    class_name_newlist.append(class_name_list[index])
            print(
                f"processing classes:{class_name_newlist}, index:{selected_class_list}"
            )
        else:

            selected_class_list = []
            # find corresponding index of the name in selected_class_name_list
            for name in self.class_list:
                selected_class_list.append(class_name_list.index(name))
            # assign selected_class_name_list to the stored class_name_list
            class_name_newlist = self.class_list
            print(
                f"processing classes:{class_name_newlist}, index:{selected_class_list}"
            )

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
        return data, index, class_name_newlist

    # prepare_data() should contains code that will be run once per dataset.
    # most of the code will be skipped in subsequent run.
    def prepare_data(self):
        """
        default function to download data / write to disk for pytorch datamodule
        now it will unzip mesh file to directory if necessary
        """

        if self.config.data_augmentation:
            # for data_augmentation:
            # put/unzip all 3D objects to a directory. Ready for setup() to read
            # after perpare_data(), the target should be a directory with all 3D object files
            if self.zip_path:
                self._unzip_zip_file_to_directory()

    # setup() should contains code that will be run once per run.
    def setup(self, stage=None):
        """
        default function to process data and assign them as attribute

        if config.data_augmentation, this call _read_mesh_array() to process the mesh data
            and then assign results to self.size and self.dataset
            if the data is conditional (num_classes > 0), also perform _trim_dataset()
            and assign results to self.size, self.dataset, self.datalabel, and self.class_list

        if not config.data_augmentation, this first check .npy/.npz file (the self.savefile_path)
            exists or not. If exists, just load the data in it and assign to self.size and self.dataset
            for .npy file , and self.size, self.dataset, self.datalabel, and self.class_list for .npz file.
            If the .npy/.npz file not exists, this still call _read_mesh_array() and _trim_dataset()
            to process the mesh data, and then save the result to .npy/.npz file. After that, load data
            from .npy/.npz file again to assign to datamodule attributes.


        Parameters
        ----------
        stage : (Default value = None)
            Not used here
        self.config.num_classes : int
            indicate maximum number of classes to read.
        self.config.data_augmentation : boolean
            whether to apply data_augmentation or not
        self.size : int
            will record the number of samples after this function runs
        self.dataset : list/numpy.ndarray
            will contain the processed mesh files after this function runs
            if data_augmentation=True, mesh files will be processed into array
            if not, mesh files just appended in list
        self.datalabel : list
            will contain the class indices of the loaded mesh files
        self.class_list : list
            will contain the class name of the loaded mesh files
            self.class_list[self.datalabel[i]] for self.dataset[i]
        """

        if self.config.data_augmentation:

            if self.num_classes is None:
                # read uncondtional data
                dataset, failed = self._read_mesh_array(
                    isZip=False, process_to_array=False
                )

                # now all the returned array contains multiple samples
                self.size = len(dataset)
                self.dataset = dataset
                print(f"Processed dataset size: {self.size}")
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

                print(f"Final dataset_array shape: {self.size}")
                print(f"Final number of classes: {self.num_classes}")
                print("class_list:", class_list)
                self.class_list = class_list

        else:
            # for normal:
            # read all files and process the object array to .npy/.npz file
            # if self.class_list is not None, selected classes from user, so reprocess again
            if self.savefile_path.exists() and (
                self.class_list is None or self.num_classes is None
            ):
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
                    # only save if do not have selected class list, so the classes are selected by the datamodule method
                    if self.class_list is None:
                        # save as .npz file for conditional data
                        np.savez(
                            self.savefile_path,
                            data=data,
                            index=index,
                            class_list=class_list,
                        )

                        print(f"Saved processed dataset to {self.savefile_path}")
                    else:
                        # as we do not save file, data are from the processed array above
                        data = np.array(data)
                        index = np.array(index)
                        self.size = data.shape[0]
                        self.dataset = torch.unsqueeze(torch.tensor(data), 1)
                        self.datalabel = torch.tensor(index)
                        self.class_list = class_list
                        return

            # load data from the processed datafile
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
        """
        default function for pytorch datamodule to return torch dataloader

        Parameters
        ----------
        self.config.num_classes : int
            indicate maximum number of classes to read.
        self.config.data_augmentation : boolean
            whether to apply data_augmentation or not

        Returns
        -------
        torch.utils.data.DataLoader
            the dataloader of the processed data
            if data_augmentation=True, dataloader with AugmentationDataset will be returned
            if num_classes > 0, the dataset also contains data label.
        """
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
                aug_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=os.cpu_count(),
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
                num_workers=os.cpu_count(),
            )

    def sample_data(self, num_samples=1):
        """
        default function for pytorch datamodule to return torch dataloader
        sample_data from dataset is used for wandbLog and calculate statistics in between epochs

        Parameters
        ----------
        self.config.num_classes : int
            indicate maximum number of classes to read.
        self.config.data_augmentation : boolean
            whether to apply data_augmentation or not

        Returns
        -------
        data : numpy ndarray
            the input mesh data from the datamodule scaled to [-1,1]
                where array is in shape (B, res, res, res)
                B = config.batch_size, res = config.resolution
            see datamodulePL.py datamodule_process class
        label : numpy ndarray
            the input data indices for conditional data from the datamodule
                where index is in shape (B,), each element is
                the class index based on the datamodule class_list
                None if the data/model is unconditional
        """
        num_data = len(self.dataset)
        if num_samples > num_data:
            num_samples = num_data
            indices = range(num_samples)
        else:
            indices = np.random.choice(num_data, num_samples, replace=False)
        if self.config.data_augmentation:
            data_list = []
            for n in range(num_samples):
                array = mesh2arrayCentered(
                    self.dataset[indices[n]], array_length=self.config.resolution
                )
                data_list.append(array)
            data = np.array(data_list) * 2 - 1
            if self.num_classes is None:
                return data
            else:
                label = [self.datalabel[i] for i in indices]
                label = np.array(label)
                return data, label
        else:
            data = np.array(self.dataset[indices])[:, 0, :, :, :]
            if self.num_classes is None:
                return data
            else:
                label = np.array(self.datalabel[indices])
                return data, label


#####
#   helper class
#####


class AugmentationDataset(Dataset):
    """
    torch Dataset that voxelize mesh file into numpy array on-the-fly

    When __getitem__() is called, the class object will rotate the required mesh with a random angle,
    voxelize it into numpy array, and return as torch.Tensor

    Parameters
    ----------
    config : Namespace
        dictionary of training parameters
    config.aug_rotation_type : string
        indicate the how to rotate the mesh when __getitem__() is called.
        "random rotation" : rotate mesh randomly
        "axis rotation" :  rotate the mesh on the specified axis in config.aug_rotation_axis
    config.aug_rotation_axis : (float, float, float)
        the axis of rotation when aug_rotation_type = "axis rotation"
    data_list : list
        the loaded mesh files (self.dataset) from the DataModule_process class
    datalabel : list
        the loaded label of the mesh files (self.datalabel) the DataModule_process class
    """

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
        """
        get the processed mesh based on the given index

        The returned tensor is based on self.data_list[index]
        The data will first be rotated based on the self.rotation_type and config.aug_rotation_axis,
            and processed into numpy array, and then the torch tensor of the array.
        If self.datalabel is not None, also return torch tensor of self.datalabel[index]

        Parameters
        ----------
        self.aug_rotation_type : string
            indicate the how to rotate the mesh when __getitem__() is called.
            "random rotation" : rotate mesh randomly
            "axis rotation" :  rotate the mesh on the specified axis in config.aug_rotation_axis
        config.aug_rotation_axis : (float, float, float)
            the axis of rotation when aug_rotation_type = "axis rotation"
        config.resolution : int
            the size of the array for voxelization
            if resolution=32, the resulting array of processing N mesh files are (N, 1, 32, 32, 32)
        """
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
        # assume selectedItem is Trimesh object
        array = mesh2arrayCentered(selectedItem, array_length=self.config.resolution)
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
def make_processed_savefile_path(path: Path, res, max_num_classes=None, offset=None):
    """
    The function to generate the path of .npy/.npz file for DataModule_process class

    for unconditional data (max_num_classes=None), .npy file path will be generated,
        else, .npz file will be generated.
    The generated path will be based on the path input (usually the data_path in DataModule_process),
        and also add resolution (res) and number of classes (max_num_classes) information
        to the file name of the .npy/.npz file

    Parameters
    ----------
    path : Path
        the path (usually the data_path in DataModule_process) that the generated .npy/.npz file
        based on.
    res : int
        the resolution, the size of the array for voxelization in DataModule_process
    max_num_classes : int
        the maximum number of classes specified in the config. Indicate the maximum number of classes to read
    """

    # TODO
    # Preferably we would not save into the dataset directory it can break
    # code that relies on there not being extra files in the dataset directory.
    #
    # We could use
    #
    #     path.parent / "{path.name}.npy"
    #
    # instead or save to an entirely different location.
    filename_tail = f"_res{res}"
    filename_suffix = ".npy"
    if max_num_classes is not None:
        filename_tail = filename_tail + f"_c{max_num_classes}"
        filename_suffix = ".npz"

    if offset is not None:
        filename_tail = filename_tail + f"_offset{offset}"

    if path.is_dir():
        return path / (f"dataset_array_processed" + filename_tail + filename_suffix)
    elif path.suffix == ".zip":
        return path.parent / (f"{path.stem}" + filename_tail + filename_suffix)
    elif path.suffix == filename_suffix:
        return path
    else:
        raise ValueError(f"Cannot handle dataset path {path}")
