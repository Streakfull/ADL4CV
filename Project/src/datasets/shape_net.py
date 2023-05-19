from torch.utils.data import Dataset
import torch
import os
from utils.sdf_reader import read_sdf
from dataset_preprocessing.constants import SDF_EXTENSION, SDF_SUFFIX


class ShapenetDataset(Dataset):
    def __init__(self, shape_dir, full_data_set_dir, resolution=64, transform=None, cat="all"):
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
        self.shapes = self.get_directory_ids()

    def __len__(self):
        return len(self.shapes)

    def get_directory_ids(self):
        if (self.cat == "all"):
            return os.listdir(self.shape_dir)
        directory_name = f"{self.full_data_set_dir}/{self.cat}/parsed_trees"
        return os.listdir(directory_name)

    def full_file_path(self, shape_id):
        """Returns the full file path of a given shape id

        Args:
            shape_id (string): The shape id (folder name)

        Returns:
            string: full folder path relative to current working directory
        """
        filename = f"{self.shape_dir}/{shape_id}/{shape_id}{SDF_SUFFIX}{SDF_EXTENSION}"
        return filename

    def __getitem__(self, idx):
        shape_id = self.shapes[idx]
        filename = self.full_file_path(shape_id)
        sdf_grid = read_sdf(filename, self.resolution)
        if self.transform:
            sdf_grid = self.transform(sdf_grid)
        sdf_grid = torch.Tensor(sdf_grid)
        return sdf_grid.view(1, self.resolution, self.resolution, self.resolution)
