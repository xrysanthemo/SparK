# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms, ToTensor
from torch.utils.data import Dataset
from typing import Optional, Union, Callable, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import torchvision
import pandas as pd
import pytorch_lightning as pl
# from pl_bolts.transforms.self_supervised.simclr_transforms import SimCLRTrainDataTransform
# import torch
# import torchvision.transforms as T
from skimage import io
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import ToPILImage
from torchvision import transforms


try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC

def get_train_transform():

    train_transform = A.Compose(
        [
            ## COMPLEX
            A.Flip(p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),

            # A.OneOf([
            #    A.OpticalDistortion(distort_limit=1.0),
            #    A.GridDistortion(num_steps=5, distort_limit=1.),
            #    A.ElasticTransform(alpha=3),
            # ], p=0.7),

            A.CLAHE(clip_limit=4.0, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                               rotate_limit=15, border_mode=0, p=0.85),

            ## SIMPLE
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
            #     rotate_limit=360, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            # A.RandomBrightnessContrast(p=0.5),

            ## WITHOUT BLUR

            # A.Flip(p=0.5),
            #
            # A.GaussNoise(var_limit=(5.0, 30.0), p=0.7),
            #
            # A.OneOf([
            #    A.OpticalDistortion(distort_limit=1.0),
            #    A.GridDistortion(num_steps=5, distort_limit=1.),
            #    # A.ElasticTransform(alpha=3),
            # ], p=0.7),
            #
            # A.CLAHE(clip_limit=4.0, p=0.7),
            # A.HueSaturationValue(hue_shift_limit=10,
            #                      sat_shift_limit=20, val_shift_limit=10, p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
            #                    rotate_limit=15, border_mode=0, p=0.85),

            ToTensorV2(),

        ]
    )
    return train_transform

class SSLDataset(Dataset):
    id_column = "isic_id"
    ground_truth_column = "diagnosis"
    def exclude_datasets(self, df):
        datasets_path = glob.glob(self.data_dir + '/excluded_datasets/*.csv')
        datasets = []
        for dataset in datasets_path:
            dataset_df = pd.read_csv(dataset)
            datasets.append(dataset_df)
        excluded_df = pd.concat(datasets)
        # #check for duplicates
        # duplicate = excluded_df[excluded_df.duplicated(subset=['isic_id'])]
        # exclude isic_ids in df from excluded_df
        df = df[~df.isic_id.isin(excluded_df.isic_id)]
        return df
    def __init__(self,
                  transform=None, data_dir: str = "./data", resize_dim: int = 512,
                  batch_size: int = 512):
        super().__init__()
        pl.seed_everything(0)
        self.data_dir = data_dir
        self.raw_images_dir = data_dir + '/raw'
        self.resize_dim = resize_dim
        self.batch_size = batch_size

        ground_truth_file = self.raw_images_dir + '/metadata.csv'  # TODO transform this to a function
        df = pd.read_csv(ground_truth_file)
        df = self.exclude_datasets(df)  # exclude problematic datasets
        df = df[~df.image_type.isin(['overview', 'clinical'])]
        self.class_distribution = df[self.ground_truth_column].value_counts()
        self.ground_truth = df[self.ground_truth_column].unique()
        self.ground_truth_meaning = dict(zip(self.ground_truth, range(len(self.ground_truth))))
        df["label"] = df[self.ground_truth_column].map(self.ground_truth_meaning)
        gt = {}
        for i, row in df.iterrows():
            gt[row.isic_id] = row["label"]
            # try:
            #     values = row["label"]
            #     gt[row.isic_id] = values
            # except:
            #     print(i)
        self.ground_truth_dict = gt
        self.ground_truth_list = list(self.ground_truth_dict.values())

        self.ground_truth = df[self.ground_truth_column].unique()
        self.ground_truth_meaning = dict(zip(self.ground_truth, range(len(self.ground_truth))))
        # TODO maybe add the same data as validation set to inspect performance? might be constraining on memory?
        self.preprocessed_image_paths = [data_dir + "/preprocessed_" + str(resize_dim) + "/" + key + '.JPG' for key in
                                         self.ground_truth_dict.keys()]

        self.image_path_list = self.preprocessed_image_paths
        self.transform = ToTensor()

        # Extend dataset

    def __len__(self):
        return len(self.image_path_list)


    def __getitem__(self, idx):
        image_idx = self.image_path_list[idx].split("/")[-1][:-4]
        image = io.imread(self.image_path_list[idx])

        if self.transform:
            image = self.transform(image)

        # images = torch.cat([images[0], images[1]], dim=0)
        return image




def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


def build_dataset_to_pretrain(dataset_path, input_size) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    # train_transform = get_train_transform()
    
    dataset_path = os.path.abspath(dataset_path)
    for postfix in ('train', 'val'):
        if dataset_path.endswith(postfix):
            dataset_path = dataset_path[:-len(postfix)]


    dataset_train = SSLDataset(data_dir=dataset_path, #"/home/xrysanthemo/datasets/all_isic"
                               resize_dim=input_size, #512
                               batch_size=64,
                               transform=None)
    # print_transform(train_transform, '[pre-train]')

    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
