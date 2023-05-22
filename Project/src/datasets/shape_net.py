from torch.utils.data import Dataset
import torch
import os
from utils.sdf_reader import read_sdf
from dataset_preprocessing.constants import SDF_EXTENSION, SDF_SUFFIX


class ShapenetDataset(Dataset):
    def __init__(self, shape_dir, full_data_set_dir, resolution=64, transform=None, cat="all", max_dataset_size=None, trunc_thres=0.2):
        """Initializes the shape dataset class 

        Args:
            shape_dir (string): directory containing all the shapes
            full_data_set_dir (string): directory containing the text2Shape++ texts
            resolution (int, optional): resolution of sdf grides. Defaults to 64.
            transform (Transform, optional): any transformation applied. Defaults to None.
            cat (str, optional): categories to retrieve all|chairs|tables .Defaults to "all".
        """

        self.shape_dir = shape_dir
        self.transform = transform
        self.resolution = resolution
        self.cat = cat
        self.full_data_set_dir = full_data_set_dir
        self.trunc_thresh = trunc_thres
        self.shapes = self.get_directory_ids()
        self.max_dataset_size = max_dataset_size
        if (max_dataset_size):
            self.shapes = self.shapes[0:max_dataset_size]

    def __len__(self):
        return len(self.shapes)

    def get_directory_ids(self):
        if (self.cat == "all"):
            return os.listdir(self.shape_dir)
        directory_name = f"{self.full_data_set_dir}/{self.cat}"
        return os.listdir(directory_name)

    def full_file_path(self, shape_id):
        """Returns the full file path of a given shape id

        Args:
            shape_id (string): The shape id (folder name)

        Returns:
            string: full folder path relative to current working directory
        """
        filename = f"{self.shape_dir}/{self.cat}/{shape_id}/ori_sample{SDF_EXTENSION}"
        return filename

    def __getitem__(self, idx):
        shape_id = self.shapes[idx]
        filename = self.full_file_path(shape_id)
        sdf_grid = read_sdf(filename, self.resolution)
        sdf_grid = torch.Tensor(sdf_grid)
        if self.transform:
            sdf_grid = self.transform(sdf_grid)

        thres = self.trunc_thres
        if thres != 0.0:
            sdf_grid = torch.clamp(sdf_grid, min=-thres, max=thres)
        return sdf_grid.view(1, self.resolution, self.resolution, self.resolution)

    def name(self):
        return 'SDFDataset'
