from torch.utils.data import Dataset
import numpy as np
import torch
import os
import json
from dataset_preprocessing.constants import Z_SET_FILENAME, Z_SHAPE_FILENAME
from datasets.dataset_utils import sample_z_set


class ShapeNetZSets(Dataset):
    def __init__(self, shape_dir, shape_set_paths="shape_set_paths.json", cat="chairs", max_dataset_size=None, codebook_indices=512):
        self.num_classes = codebook_indices
        json_file = open(f"{shape_dir}/{shape_set_paths}")
        self.shape_set_paths_json = json.load(json_file)
        self.shape_dir = shape_dir
        self.cat = cat
        self.shape_sets = list(self.shape_set_paths_json.keys())
        self.single_shapes = os.listdir(f"{shape_dir}/{cat}")

    def __len__(self):
        return len(self.shape_sets) + len(self.single_shapes)

    # TODO: Handle this better with the shapenet dataset

    def get_shape_directory(self, shape_id):
        return f"{self.shape_dir}/{self.cat}/{shape_id}"

    def access_file(self, idx):
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

    # Override

    def __getitem__(self, idx):
        file_access = self.access_file(idx)
        z_set = torch.load(file_access, map_location=torch.device('cpu'))
        q_set = sample_z_set(z_set)
        return {
            "z_set": z_set,
            "q_set": q_set
        }
