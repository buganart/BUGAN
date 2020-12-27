from argparse import Namespace
from pathlib import Path
import shutil
import zipfile

import pytest

from bugan.functionsPL import DataModule_process, DataModule_process_cond


@pytest.fixture(params=["zip", "folder"])
def data_path(request, tmp_path):
    data_folder = Path(__file__).parent / "data"
    if request.param == "folder":
        tmp_data_path = shutil.copytree(data_folder, tmp_path / "data")
        return tmp_data_path
    else:
        zip_path = shutil.make_archive(tmp_path / "dataset", "zip", data_folder)
        return zip_path


@pytest.mark.parametrize("data_augmentation", [True, False])
@pytest.mark.parametrize("rotation_type", ["random rotation", "axis rotation"])
def test_data_module_folder(data_path, data_augmentation, rotation_type):

    config = Namespace(
        batch_size=1,
        resolution=32,
        data_augmentation=data_augmentation,
        aug_rotation_type=rotation_type,
        aug_rotation_axis=(0, 1, 0),
    )

    data_module = DataModule_process(config, run=None, data_path=data_path)
    data_module.prepare_data()
    data_module.setup()

    batch = next(iter(data_module.train_dataloader()))
    expected_shape = [
        1,
        1,
        32,
        32,
        32,
    ]
    assert list(batch[0].shape) == expected_shape


@pytest.fixture(params=["zip", "folder"])
def data_path_cond(request, tmp_path):
    data_folder = Path(__file__).parent / "data"
    if request.param == "folder":
        tmp_data_path = shutil.copytree(data_folder, tmp_path / "data/class_1")
        return tmp_data_path
    else:
        zip_path = shutil.make_archive(tmp_path / "dataset/class_1", "zip", data_folder)
        return zip_path


@pytest.mark.parametrize("data_augmentation", [True, False])
@pytest.mark.parametrize("rotation_type", ["random rotation", "axis rotation"])
def test_data_module_folder_cond(data_path_cond, data_augmentation, rotation_type):

    config = Namespace(
        batch_size=1,
        resolution=32,
        data_augmentation=data_augmentation,
        aug_rotation_type=rotation_type,
        aug_rotation_axis=(0, 1, 0),
        num_classes=1,
    )

    data_module = DataModule_process_cond(config, run=None, data_path=data_path_cond)
    data_module.prepare_data()
    data_module.setup()

    batch = next(iter(data_module.train_dataloader()))
    mesh, index = batch
    expected_shape = [
        1,
        1,
        32,
        32,
        32,
    ]
    assert list(mesh.shape) == expected_shape
