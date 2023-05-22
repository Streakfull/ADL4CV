import pdb
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', '')))  # noqa
from dataset_preprocessing.constants import DATA_SET_PATH, TEXT2SHAPE_DATA_SET_PATH
from datasets.latent_code_extractor import LatentCodeExtractor


root = ".."
dataset_path = f"{root}/{DATA_SET_PATH}"
text2Shape_dataset_path = f"{root}/{TEXT2SHAPE_DATA_SET_PATH}"

dataset = LatentCodeExtractor(
    text2Shape_dataset_path, dataset_path, cat="chairs", resolution=32)

# x = latent_code_dataset[0]

pdb.set_trace()
# print(x, "XXX")
