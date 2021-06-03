from argparse import Namespace
from pathlib import Path
import shutil
import zipfile

import pytest

from bugan.datamodulePL import DataModule_process

# handle Error #15: Initializing libiomp5md.dll ......
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@pytest.fixture(params=["zip", "folder"])
def data_process_format(request):
    return request.param


@pytest.fixture(params=[True, False])
def isConditionalData(request):
    return request.param


@pytest.fixture
def data_path(data_process_format, isConditionalData, tmp_path):
    data_folder = Path(__file__).parent / "data"
    if isConditionalData:
        localDirPath = "data/class_1"
    else:
        localDirPath = "data"

    if data_process_format == "folder":
        tmp_data_path = shutil.copytree(data_folder, tmp_path / localDirPath)
        return tmp_data_path
    else:
        zip_path = shutil.make_archive(tmp_path / localDirPath, "zip", data_folder)
        return zip_path


@pytest.mark.parametrize("data_augmentation", [True, False])
@pytest.mark.parametrize("rotation_type", ["random rotation", "axis rotation"])
def test_data_module_folder(
    data_path, data_augmentation, rotation_type, isConditionalData
):
    # separate isConditionalData from data_path
    # data_path, isConditionalData = data_path

    config = Namespace(
        batch_size=1,
        resolution=32,
        data_augmentation=data_augmentation,
        aug_rotation_type=rotation_type,
        aug_rotation_axis=(0, 1, 0),
    )

    if isConditionalData:
        # add num_classes into config
        config.num_classes = 1

    data_module = DataModule_process(config, data_path=data_path)
    data_module.prepare_data()
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    if isConditionalData:
        # batch from conditional data contains index
        mesh, index = batch
    else:
        mesh = batch[0]

    expected_shape = [
        1,
        1,
        32,
        32,
        32,
    ]
    assert list(mesh.shape) == expected_shape
