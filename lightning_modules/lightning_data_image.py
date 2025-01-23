import torchvision.transforms as transforms
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from torch.utils.data import Subset
import torchvision
import lightning
import os
from argparse import Namespace

from dataset.videodataset import VideoClsDataset

from dataset.randomaug import RandAugment


def MaybeGetDistributedSampler():
    if torch.distributed.is_initialized():
        return DistributedSampler
    return RandomSampler


def split_by_rank(dataset, rank, world_size):
    len_valset = len(dataset)
    split_size = len_valset // world_size
    return Subset(dataset, range(rank * split_size, (rank + 1) * split_size))


class BaseData(lightning.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        per_device_batch_size: int = 32,
        imsize: int | list = 224,
        num_workers: int = 6,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.per_device_batch_size = per_device_batch_size
        self.save_hyperparameters()
        self.trainset = None
        self.valset = None
        self.imsize = imsize

    def setup(self, stage: str):
        NotImplementedError("Please setup the dataset")
        pass

    def train_dataloader(self):
        sampler_train = DistributedSampler(
            self.trainset,
            # num_replicas=dist.get_world_size(),
            # rank=dist.get_rank(),
        )
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            sampler=sampler_train,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=3,
            drop_last=True,
        )

    def val_dataloader(self):

        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=3,
            drop_last=True,
        )


# Cifar 100
class Cifar100(BaseData):
    def __init__(
        self,
        data_dir: str = "./data",
        per_device_batch_size: int = 64,
        imsize: int = 32,
        num_workers: int = 6,
        ablation_factor: float = 1.0,
    ):
        super().__init__(
            data_dir=data_dir,
            per_device_batch_size=per_device_batch_size,
            imsize=imsize,
            num_workers=num_workers,
        )

        self.transform_train = transforms.Compose(
            [
                # transforms.RandAugment(num_ops=2, magnitude=14),
                RandAugment(2, 14),
                transforms.Resize(imsize),  # in case we do
                transforms.RandomCrop(imsize, padding=4),
                transforms.Resize(imsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize(imsize),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.ablation_factor = ablation_factor
        self.save_hyperparameters()

    def setup(self, stage: str):
        self.trainset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform_train,
        )
        if self.ablation_factor < 1:
            generator = torch.Generator().manual_seed(42)
            num_samples = int(len(self.trainset) * self.ablation_factor)
            indices = torch.randperm(len(self.trainset), generator=generator)[
                :num_samples
            ].tolist()
            self.trainset = Subset(self.trainset, indices)

        full_valset = torchvision.datasets.CIFAR100(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform_test,
        )  # use the test set for validation since we don't use the test set.
        self.full_valset = full_valset

        if self.trainer == None:
            global_rank = 0
            world_size = 1
        else:
            global_rank = self.trainer.global_rank
            world_size = self.trainer.world_size

        self.valset = split_by_rank(
            full_valset,
            rank=global_rank,
            world_size=world_size,
        )

    def train_dataloader(self):
        sampler_train = DistributedSampler(
            self.trainset,
        )
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            sampler=sampler_train,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=3,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.per_device_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=3,
        )


class Imagenet(Cifar100):
    def __init__(self, data_dir=".", per_device_batch_size=58, imsize=224):
        super().__init__(
            data_dir=data_dir,
            per_device_batch_size=per_device_batch_size,
            imsize=imsize,
        )

        self.transform_train = transforms.Compose(
            [
                RandAugment(2, 14),
                transforms.RandomResizedCrop(imsize),
                # transforms.RandomCrop(imsize, padding=imsize//8),
                # transforms.Resize(imsize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.CenterCrop(imsize),  # Is this right?
                transforms.Resize(imsize),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def setup(self, stage: str):
        self.trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"), transform=self.transform_train
        )
        full_valset = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"), transform=self.transform_test
        )

        self.valset = split_by_rank(
            full_valset,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
        )

class UCF101_fast(BaseData):
    def setup(self, stage):
        anno_path = os.path.join(self.data_dir, "annotations")
        data_path = os.path.join(self.data_dir, "data")
        config = Namespace()
        config.train_interpolation = "bicubic"
        config.reprob = 0.25
        config.num_sample = 0
        config.aa = "rand-m7-n4-mstd0.5-inc1"
        config.remode = "pixel"
        config.recount = 1
        # config.fc_drop_rate = 0.5
        config.data_set = "ucf101"

        self.trainset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode="train",
            clip_len=self.imsize[0],
            frame_sample_rate=4,
            num_segment=1,
            test_num_segment=10,
            test_num_crop=3,
            keep_aspect_ratio=True,
            crop_size=224,
            short_side_size=224,
            new_height=256,
            new_width=320,
            args=config,
        )

        full_valset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode="test",
            clip_len=self.imsize[0],
            frame_sample_rate=4,
            num_segment=1,
            test_num_segment=5,
            test_num_crop=3,
            keep_aspect_ratio=True,
            crop_size=224,
            short_side_size=224,
            new_height=256,
            new_width=320,
            args=config,
        )
        self.full_valset = full_valset

        if self.trainer == None:
            global_rank = 0
            world_size = 1
        else:
            global_rank = self.trainer.global_rank
            world_size = self.trainer.world_size

        self.valset = split_by_rank(
            full_valset,
            rank=global_rank,
            world_size=world_size,
        )
        assert len(self.valset) > 0