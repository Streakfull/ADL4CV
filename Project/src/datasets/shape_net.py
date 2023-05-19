from torch.utils.data import Dataset
import torch
import os
from utils.sdf_reader import read_sdf
from dataset_preprocessing.constants import SDF_EXTENSION, SDF_SUFFIX


class ShapenetDataset(Dataset):
    def __init__(self, shape_dir, resolution, transform=None, cat="all"):
        self.shape_dir = shape_dir
        self.transform = transform
        self.shapes = os.listdir(self.shape_dir)
        self.resolution = resolution

    def __len__(self):
        return len(self.shapes)

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
