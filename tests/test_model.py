from argparse import Namespace

import numpy as np
import pytest
import torch
import wandb
import pytorch_lightning as pl

from bugan.modelsPL import VAEGAN, VAE_train, GAN, GAN_Wloss, GAN_Wloss_GP, CGAN
from bugan.functionsPL import DataModule_process, DataModule_process_cond
from test_data_loader import data_path_cond, data_path


@pytest.fixture
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def wandb_init_run():
    run = wandb.init(
        project="tree-gan",
        id=wandb.util.generate_id(),
        entity="bugan",
        anonymous="allow",
        mode="offline",
        dir="../",
    )
    return run


### CHECK FORWARD


def test_vaegan_forward(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
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


def test_vae_forward(device):
    config = Namespace(
        resolution=32,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        vae_decoder_layer=1,
        vae_encoder_layer=1,
        z_size=2,
    )
    model = VAE_train(config).to(device)
    x = torch.tensor(np.ones([2, 1, 32, 32, 32], dtype=np.float32)).to(device)
    y = model(x)
    assert list(y.shape) == [2, 1, 32, 32, 32]


def test_gan_forward(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
    )
    model = GAN(config).to(device)
    x = torch.tensor(np.ones([2, 2], dtype=np.float32)).to(device)
    y = model(x)
    assert list(y.shape) == [2, 1]


def test_gan_wloss_forward(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
    )
    model = GAN_Wloss(config).to(device)
    x = torch.tensor(np.ones([2, 2], dtype=np.float32)).to(device)
    y = model(x)
    assert list(y.shape) == [2, 1]


def test_gan_wloss_gp_forward(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
    )
    model = GAN_Wloss_GP(config).to(device)
    x = torch.tensor(np.ones([2, 2], dtype=np.float32)).to(device)
    y = model(x)
    assert list(y.shape) == [2, 1]


def test_cgan_forward(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
        num_classes=3,
    )
    model = CGAN(config).to(device)
    x = torch.tensor(np.ones([2, 2], dtype=np.float32)).to(device)
    c = torch.tensor(np.ones([2], dtype=np.int64)).to(device)
    y, c_predict = model(x, c)
    assert list(y.shape) == [2, 1]
    assert list(c_predict.shape) == [2, config.num_classes]


### CHECK TRAINING STEP


def test_vaegan_training_step(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        vae_decoder_layer=1,
        vae_encoder_layer=1,
        z_size=2,
    )
    model = VAEGAN(config).to(device)
    data = torch.tensor(np.ones([2, 1, 32, 32, 32], dtype=np.float32)).to(device)
    model.on_train_epoch_start()
    loss_vae = model.training_step(dataset_batch=[data], batch_idx=0, optimizer_idx=0)
    loss_d = model.training_step(dataset_batch=[data], batch_idx=0, optimizer_idx=1)
    # tensor with single element has shape []
    assert list(loss_vae.shape) == [] and not loss_vae.isnan() and not loss_vae.isinf()
    assert list(loss_d.shape) == [] and not loss_d.isnan() and not loss_d.isinf()


def test_vae_training_step(device):
    config = Namespace(
        resolution=32,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        vae_decoder_layer=1,
        vae_encoder_layer=1,
        z_size=2,
    )
    model = VAE_train(config).to(device)
    data = torch.tensor(np.ones([2, 1, 32, 32, 32], dtype=np.float32)).to(device)
    model.on_train_epoch_start()
    loss_vae = model.training_step(dataset_batch=[data], batch_idx=0)
    # tensor with single element has shape []
    assert list(loss_vae.shape) == [] and not loss_vae.isnan() and not loss_vae.isinf()


def test_gan_training_step(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
        batch_size=2,
    )
    model = GAN(config).to(device)
    data = torch.tensor(np.ones([2, 1, 32, 32, 32], dtype=np.float32)).to(device)
    model.on_train_epoch_start()
    loss_g = model.training_step(dataset_batch=[data], batch_idx=0, optimizer_idx=0)
    loss_d = model.training_step(dataset_batch=[data], batch_idx=0, optimizer_idx=1)
    # tensor with single element has shape []
    assert list(loss_g.shape) == [] and not loss_g.isnan() and not loss_g.isinf()
    assert list(loss_d.shape) == [] and not loss_d.isnan() and not loss_d.isinf()


def test_gan_wloss_training_step(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
    )
    model = GAN_Wloss(config).to(device)
    data = torch.tensor(np.ones([2, 1, 32, 32, 32], dtype=np.float32)).to(device)
    model.on_train_epoch_start()
    loss_g = model.training_step(dataset_batch=[data], batch_idx=0, optimizer_idx=0)
    loss_d = model.training_step(dataset_batch=[data], batch_idx=0, optimizer_idx=1)
    # tensor with single element has shape []
    assert list(loss_g.shape) == [] and not loss_g.isnan() and not loss_g.isinf()
    assert list(loss_d.shape) == [] and not loss_d.isnan() and not loss_d.isinf()


def test_gan_wloss_gp_training_step(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
    )
    model = GAN_Wloss_GP(config).to(device)
    data = torch.tensor(np.ones([2, 1, 32, 32, 32], dtype=np.float32)).to(device)
    model.on_train_epoch_start()
    loss_g = model.training_step(dataset_batch=[data], batch_idx=0, optimizer_idx=0)
    loss_d = model.training_step(dataset_batch=[data], batch_idx=0, optimizer_idx=1)
    # tensor with single element has shape []
    assert list(loss_g.shape) == [] and not loss_g.isnan() and not loss_g.isinf()
    assert list(loss_d.shape) == [] and not loss_d.isnan() and not loss_d.isinf()


def test_cgan_training_step(device):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
        num_classes=3,
    )
    model = CGAN(config).to(device)
    data = torch.tensor(np.ones([2, 1, 32, 32, 32], dtype=np.float32)).to(device)
    label = torch.tensor(np.ones([2], dtype=np.int64)).to(device)
    model.on_train_epoch_start()
    loss_g = model.training_step(
        dataset_batch=[data, label], batch_idx=0, optimizer_idx=0
    )
    loss_d = model.training_step(
        dataset_batch=[data, label], batch_idx=0, optimizer_idx=1
    )
    loss_c = model.training_step(
        dataset_batch=[data, label], batch_idx=0, optimizer_idx=2
    )
    # tensor with single element has shape []
    assert list(loss_g.shape) == [] and not loss_g.isnan() and not loss_g.isinf()
    assert list(loss_d.shape) == [] and not loss_d.isnan() and not loss_d.isinf()
    assert list(loss_c.shape) == [] and not loss_c.isnan() and not loss_c.isinf()


### CHECK FULL TRAINING (with data_module, model, and trainer)


def test_vaegan_training_loop_full(device, wandb_init_run, data_path):
    config = Namespace(
        resolution=32,
        d_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        vae_decoder_layer=1,
        vae_encoder_layer=1,
        z_size=2,
        # for dataloader
        batch_size=1,
        data_augmentation=True,
        aug_rotation_type="random rotation",
        aug_rotation_axis=(0, 1, 0),
    )
    model = VAEGAN(config).to(device)
    data_module = DataModule_process(config, run=wandb_init_run, data_path=data_path)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, data_module)


def test_vae_training_loop_full(device, wandb_init_run, data_path):
    config = Namespace(
        resolution=32,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        vae_decoder_layer=1,
        vae_encoder_layer=1,
        z_size=2,
        # for dataloader
        batch_size=1,
        data_augmentation=True,
        aug_rotation_type="random rotation",
        aug_rotation_axis=(0, 1, 0),
    )
    model = VAE_train(config).to(device)
    data_module = DataModule_process(config, run=wandb_init_run, data_path=data_path)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, data_module)


def test_gan_training_loop_full(device, wandb_init_run, data_path):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
        # for dataloader
        batch_size=1,
        data_augmentation=True,
        aug_rotation_type="random rotation",
        aug_rotation_axis=(0, 1, 0),
    )
    model = GAN(config).to(device)
    data_module = DataModule_process(config, run=wandb_init_run, data_path=data_path)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, data_module)


def test_gan_wloss_training_loop_full(device, wandb_init_run, data_path):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
        # for dataloader
        batch_size=1,
        data_augmentation=True,
        aug_rotation_type="random rotation",
        aug_rotation_axis=(0, 1, 0),
    )
    model = GAN_Wloss(config).to(device)
    data_module = DataModule_process(config, run=wandb_init_run, data_path=data_path)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, data_module)


def test_gan_wloss_gp_training_loop_full(device, wandb_init_run, data_path):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
        # for dataloader
        batch_size=1,
        data_augmentation=True,
        aug_rotation_type="random rotation",
        aug_rotation_axis=(0, 1, 0),
    )
    model = GAN_Wloss_GP(config).to(device)
    data_module = DataModule_process(config, run=wandb_init_run, data_path=data_path)
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, data_module)


def test_cgan_training_loop_full(device, wandb_init_run, data_path_cond):
    config = Namespace(
        resolution=32,
        d_layer=1,
        g_layer=1,
        gen_num_layer_unit=[2, 2, 2, 2],
        dis_num_layer_unit=[2, 2, 2, 2],
        z_size=2,
        # for dataloader
        batch_size=1,
        data_augmentation=True,
        aug_rotation_type="random rotation",
        aug_rotation_axis=(0, 1, 0),
        num_classes=1,
    )
    model = CGAN(config).to(device)
    data_module = DataModule_process_cond(
        config, run=wandb_init_run, data_path=data_path_cond
    )
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, data_module)
