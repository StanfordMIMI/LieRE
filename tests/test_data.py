# run with 
# MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 pytest tests/test_data.py -v
import pytest
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.videodataset import VideoClsDataset
from lightning_modules.lightning_data_image import Cifar100, Imagenet, UCF101_fast

PATH_UCF101 = "/dataNAS/people/sostm/data/sostm_ucf101"
PATH_IMAGENET = "/dataNAS/people/sostm/data/gcp_imagenet/sostm_imagenet"

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    if not dist.is_initialized():
        dist.init_process_group('gloo', init_method='env://', rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()

@pytest.fixture(scope="session")
def pl_trainer():
    return pl.Trainer(accelerator='cpu', devices=1, strategy='ddp_spawn')

@pytest.fixture
def cifar100_datamodule():
    return Cifar100(per_device_batch_size=8, imsize=32)

@pytest.fixture
def imagenet_datamodule():
    return Imagenet(data_dir=PATH_IMAGENET, per_device_batch_size=8, imsize=224)

@pytest.fixture
def ucf101_datamodule():
    return UCF101_fast(data_dir=PATH_UCF101, per_device_batch_size=8, imsize=[16])

def test_cifar100_setup(cifar100_datamodule, pl_trainer, setup_env):
    cifar100_datamodule.trainer = pl_trainer
    cifar100_datamodule.setup(stage="fit")
    assert cifar100_datamodule.trainset is not None
    assert cifar100_datamodule.valset is not None
    assert len(cifar100_datamodule.valset) > 0

def test_cifar100_dataloader(cifar100_datamodule, pl_trainer, setup_env):
    cifar100_datamodule.trainer = pl_trainer
    cifar100_datamodule.setup(stage="fit")
    train_loader = cifar100_datamodule.train_dataloader()
    val_loader = cifar100_datamodule.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    train_batch = next(iter(train_loader))
    assert isinstance(train_batch[0], torch.Tensor)
    assert train_batch[0].shape[1:] == (3, 32, 32)

def test_imagenet_setup(imagenet_datamodule, pl_trainer, setup_env):
    imagenet_datamodule.trainer = pl_trainer
    imagenet_datamodule.setup(stage="fit")
    assert imagenet_datamodule.trainset is not None
    assert imagenet_datamodule.valset is not None
    assert len(imagenet_datamodule.valset) == 50000
    assert len(imagenet_datamodule.trainset) == 1281167

def test_imagenet_dataloader(imagenet_datamodule, pl_trainer, setup_env):
    imagenet_datamodule.trainer = pl_trainer
    imagenet_datamodule.setup(stage="fit")
    train_loader = imagenet_datamodule.train_dataloader()
    val_loader = imagenet_datamodule.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    train_batch = next(iter(train_loader))
    assert isinstance(train_batch[0], torch.Tensor)

def test_ucf101_setup(ucf101_datamodule, pl_trainer, setup_env):
    ucf101_datamodule.trainer = pl_trainer
    ucf101_datamodule.setup(stage="fit")
    assert ucf101_datamodule.trainset is not None
    assert ucf101_datamodule.valset is not None
    assert len(ucf101_datamodule.valset) > 0

def test_ucf101_dataloader(ucf101_datamodule, pl_trainer, setup_env):
    ucf101_datamodule.trainer = pl_trainer
    ucf101_datamodule.setup(stage="fit")
    train_loader = ucf101_datamodule.train_dataloader()
    val_loader = ucf101_datamodule.val_dataloader()

    assert isinstance(train_loader, DataLoader), print("train_loader: ", type(train_loader))
    assert isinstance(val_loader, DataLoader), print("val_loader: ", type(val_loader))

    train_batch = next(iter(train_loader))
    assert isinstance(train_batch[0], torch.Tensor)
