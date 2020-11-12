from argparse import Namespace

import numpy as np
import pytest
import torch

from bugan.modelsPL import VAEGAN


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_vaegan_forward(device):
    config = Namespace(
        array_size=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        vae_decoder_layer=1,
        vae_encoder_layer=1,
        z_size=2,
    )
    model = VAEGAN(config).to(device)
    x = torch.tensor(np.ones([2, 1, 32, 32, 32], dtype=np.float32)).to(device)
    y = model(x)
    assert list(y.shape) == [2, 1]
