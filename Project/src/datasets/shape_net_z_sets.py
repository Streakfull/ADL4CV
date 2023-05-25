from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
from dataset_preprocessing.constants import Z_SHAPE_FILENAME
from datasets.dataset_utils import sample_z_set


class ShapeNetZSets(Dataset):
    def __init__(self, shape_dir, shape_set_paths="shape_set_paths.json", cat="chairs", max_data_set_size=None):
        """Aggregation dataset of all z_shapes and z_sets

        Args:
            shape_dir (string): directory containing all the shapes
            shape_set_paths (dict {shape_set_filename: directory }, optional): dict containing all the shapesets saved . Defaults to "shape_set_paths.json".
            cat (str chairs|tables , optional): category of shapenet shape . Defaults to "chairs".
        """
        json_file = open(f"{shape_dir}/{shape_set_paths}")
        self.shape_set_paths_json = json.load(json_file)
        self.shape_dir = shape_dir
        self.cat = cat
        self.shape_sets = list(self.shape_set_paths_json.keys())
        self.single_shapes = os.listdir(f"{shape_dir}/{cat}")
        self.max_data_size = max_data_set_size

    def __len__(self):
        if (self.max_data_size != None):
            return self.max_data_size
        return len(self.shape_sets) + len(self.single_shapes)

    # TODO: Handle this better with the shapenet dataset

    def get_shape_directory(self, shape_id):
        """returns full path of the shape directory belonging to the shape_id

        Args:
            shape_id string: current shape id

        Returns:
            string: full path of the shape directory
        """
        return f"{self.shape_dir}/{self.cat}/{shape_id}"

    def access_file(self, idx):
        """Determines how to access a file based on index, an index can either corresspond
           to a shape set or a single shape.

        Args:
            idx (int): index of the given element

        Returns:
           filename: location of the tensor on disk
        """
        if (idx < len(self.shape_sets)):
            file_name = self.shape_sets[idx]
            z_set_shape_id = self.shape_set_paths_json[file_name]
            directory = self.get_shape_directory(z_set_shape_id)
            return f"{directory}/{file_name}"
        else:
            adjusted_index = idx - len(self.shape_sets)
            shape_id = self.single_shapes[adjusted_index]
            directory = self.get_shape_directory(shape_id)
            filename = Z_SHAPE_FILENAME
            return f"{directory}/{filename}"

    def __getitem__(self, idx):
        """Gets an item from the dataset

        Args:
            idx (int): index of the given element

        Returns:
            {z_set, q_set}: one hot encoded representations g^3 * codebook indices,
                            z_set is the input data and q_set is the target vector  
        """
        file_access = self.access_file(idx)
        z_set = torch.load(file_access, map_location=torch.device('cpu'))
        q_set = sample_z_set(z_set)
        return {
            "z_set": z_set,
            "q_set": q_set,
            "idx": q_set
        }
