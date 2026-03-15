# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import numpy as np

from logging import getLogger
from PIL import Image

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False)
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info('ImageNet dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler


def make_flyingObjects3d(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
    target_type='disparity',
    pass_name='clean',
    max_samples=2000
):
    """
    Create FlyingThings3D stereo dataset and dataloader.

    Matches I-JEPA's make_imagenet1k signature for compatibility.

    Args:
        transform: Transform to apply to images
        batch_size: Batch size
        collator: MaskCollator for generating masks
        pin_mem: Pin memory for DataLoader
        num_workers: Number of workers
        world_size: Number of GPUs (1 for single GPU)
        rank: GPU rank (0 for single GPU)
        root_path: Path to FlyingThings3D subset root directory
        training: Whether to use train or test split
        drop_last: Drop last incomplete batch
        target_type: 'disparity', 'flow', or None
        pass_name: 'clean' or 'final'
        max_samples: Maximum number of stereo pairs to use

    Returns:
        dataset, data_loader, sampler (for compatibility with I-JEPA)
    """
    split = 'train' if training else 'val'

    # Use custom FlyingThings3D subset dataset
    dataset = FlyingThings3DSubset(
        root=root_path,
        split=split,
        pass_name=pass_name,
        transform=transform,
        max_samples=max_samples
    )

    logger.info(
        f'FlyingThings3D dataset created ({split} split, {len(dataset)} samples)')

    # For single GPU, use regular sampler (not distributed)
    if world_size == 1:
        sampler = torch.utils.data.RandomSampler(dataset) if training else None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=(sampler is None and training),  # Only shuffle if no sampler
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('FlyingThings3D data loader created')

    return dataset, data_loader, sampler


class FlyingThings3DSubset(torch.utils.data.Dataset):
    """
    Custom dataset for FlyingThings3D subset.

    Folder structure expected:
    root/
        train/
            image_clean/
                left/
                    0000000.png
                    0000001.png
                    ...
                right/
                    0000000.png
                    0000001.png
                    ...
        val/
            image_clean/
                left/
                right/
    """

    def __init__(self, root, split='train', pass_name='clean', transform=None, max_samples=2000):
        """
        Args:
            root: Path to FlyingThings3D subset root directory
            split: 'train' or 'val'
            pass_name: 'clean' or 'final' (folder is named image_clean or image_final)
            transform: Transform to apply to images
            max_samples: Maximum number of stereo pairs to keep
        """
        self.root = root
        self.split = split
        self.pass_name = pass_name
        self.transform = transform
        self.max_samples = max_samples

        # Construct paths based on subset structure
        image_folder = f'image_{pass_name}'
        self.left_dir = os.path.join(root, split, image_folder, 'left')
        self.right_dir = os.path.join(root, split, image_folder, 'right')

        # Verify directories exist
        if not os.path.exists(self.left_dir):
            raise FileNotFoundError(
                f"Left image directory not found: {self.left_dir}\n"
                f"Expected structure: {root}/{split}/{image_folder}/left/"
            )
        if not os.path.exists(self.right_dir):
            raise FileNotFoundError(
                f"Right image directory not found: {self.right_dir}\n"
                f"Expected structure: {root}/{split}/{image_folder}/right/"
            )

        # Get list of image files (assuming left and right have same filenames)
        self.image_files = sorted([
            f for f in os.listdir(self.left_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.left_dir}")

        if self.max_samples is not None:
            original_count = len(self.image_files)
            self.image_files = self.image_files[:self.max_samples]
            if len(self.image_files) < original_count:
                logger.info(
                    f"FlyingThings3DSubset: Limiting {split} split to {len(self.image_files)} stereo pairs "
                    f"(from {original_count})")

        logger.info(
            f"FlyingThings3DSubset: Found {len(self.image_files)} stereo pairs in {split} split")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # Load left and right images
        left_path = os.path.join(self.left_dir, img_name)
        right_path = os.path.join(self.right_dir, img_name)

        img_left = Image.open(left_path).convert('RGB')
        img_right = Image.open(right_path).convert('RGB')

        # Apply same transform to both views with same random seed
        if self.transform is not None:
            seed = np.random.randint(2147483647)

            torch.manual_seed(seed)
            np.random.seed(seed)
            img_left = self.transform(img_left)

            torch.manual_seed(seed)
            np.random.seed(seed)
            img_right = self.transform(img_right)

        # Return stereo pair
        return (img_left, img_right)


class StereoDatasetWrapper(torch.utils.data.Dataset):
    """
    Wrapper for FlyingThings3D to work with I-JEPA's MaskCollator.

    FlyingThings3D returns: (img_left, img_right, target)
    This wrapper returns: (img_left, img_right) for stereo training
    """

    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_left, img_right, target = self.base_dataset[idx]

        # Apply same transform to both views
        if self.transform is not None:
            # Use same random seed for both to ensure consistent augmentation
            seed = np.random.randint(2147483647)

            torch.manual_seed(seed)
            np.random.seed(seed)
            img_left = self.transform(img_left)

            torch.manual_seed(seed)
            np.random.seed(seed)
            img_right = self.transform(img_right)

        # Return stereo pair (target/disparity can be added later for evaluation)
        return (img_left, img_right)


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_file='imagenet_full_size-061417.tar.gz',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True,
        index_targets=False
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = 'train/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(
                    self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(
                    mint, len(indices))
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}')


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_file='imagenet_full_size-061417.tar.gz',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path
