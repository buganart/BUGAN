from argparse import Namespace
from pathlib import Path

import pytest

from bugan.functionsPL import DataModule_process


@pytest.mark.parametrize("data_augmentation", [True, False])
@pytest.mark.parametrize("rotation_type", ["random rotation", "axis rotation"])
def test_data_module_folder(data_augmentation, rotation_type):

    config = Namespace(
        batch_size=1,
        array_size=32,
        data_augmentation=data_augmentation,
        aug_rotation_type="random rotation",
    )

    data_path = Path(__file__).parent / "data"
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
