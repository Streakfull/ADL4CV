import pdb
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join('..', '')))  # noqa
from dataset_preprocessing.constants import DATA_SET_PATH, TEXT2SHAPE_DATA_SET_PATH, FULL_DATA_SET_PATH
from datasets.latent_code_extractor import LatentCodeExtractor
from options.base_options import BaseOptions
from encoder.pvqae import PVQVAE
from torch.nn import functional
from tqdm import tqdm
import json
from termcolor import colored
import pyblaze.multiprocessing as xmp
import numpy as np
import ast


device = "cpu"
if (torch.cuda.is_available()):
    device = "cuda"
root = ".."
dataset_path = f"{root}/{DATA_SET_PATH}"
text2Shape_dataset_path = f"{root}/{TEXT2SHAPE_DATA_SET_PATH}"
cat = "chairs"
shapeset_length_threshold = 40

dataset = LatentCodeExtractor(
    text2Shape_dataset_path, dataset_path, cat=cat, resolution=64, load_similar_shapes=True)


vq_cfg = f"{root}/configs/pvqae_configs.yaml"
# Chckpoint here
chkpoint = f"{root}/{FULL_DATA_SET_PATH}/checkpoints/experiment_name/ckpt/vqvae_epoch-best.pth"
options = BaseOptions(config_path=vq_cfg,
                      name="pvqae-extraction", is_train=False)
model = PVQVAE()
model.initialize(options)
model.load_ckpt(chkpoint)

stats = {
    "shapes_dict": {},
    "shapes_counter": 0,
    "shape_set_counter": 0,
    "row_indices": {}
}
# codebook_indices = model.n_embed

codebook_indices = 512
n_cubes_per_dim = 8


def save_tensor(input, shape_id, is_shape_set=False, shape_set_row_index=0):
    """Saves z-shape or z-set(by row index)

    Args:
        input tensor: Tensor to be saved
        shape_id string: Shape id or incase of a shape set the reference shape id
        is_shape_set (bool, optional): Indicates if this is a shape set. Defaults to False.
        shape_set_row_index (int, optional): Shape set row index from the dataset csv. Defaults to 0.
    """
    file_path = f"{dataset_path}/{cat}/{shape_id}/z_shape.pt"
    # file_path = f"{dataset_path}/z_shape.pt"
    if is_shape_set:
        file_path = f"{dataset_path}/{cat}/{shape_id}/z_set_{shape_set_row_index}.pt"
        # file_path = f"{dataset_path}/z_set_{shape_set_row_index}.pt"
    with open(file_path, 'wb') as f:
        torch.save(input, f)
    print(colored(f'[*]{file_path} shapeset saved', 'yellow'))


def save_stats_dict():
    """Saves the stats that has all relevant info about the process in the dataset path
    """
    with open(f"{dataset_path}/stats-shapeset-4.json", 'w') as f:
        # Write the dictionary to the file in JSON format
        json.dump(stats, f)


def save_dict(file_name, dictionary):
    with open(f"{dataset_path}/{file_name}", 'w') as f:
        # Write the dictionary to the file in JSON format
        json.dump(dictionary, f)


def handle_shape(shape_id, shape_sdf):
    """ Converts a shape sdf to be a g^3 x n_embed probability distribution over the codebook indices

    Args:
        shape_id string: Shape id
        shape_sdf tensor[1,resolution,resolution,resolution]: sdf grid of the shape
    Saves:
        z tensor[ncubes_per_dim,ncubes_per_dim,ncubes_per_dim,n_embed ]: probability distribution over all possible codebook indices
    """
    # is_shape_id_done = stats["shapes_dict"].get(shape_id, False)
    # if (is_shape_id_done):
    #     return
    # 8 * 8 * 8
    # Unsequeezed since the model expects a batch dimension
    indices = model.encode_indices(shape_sdf.unsqueeze(0)).squeeze(0)
    # One hot encode
    z = functional.one_hot(indices, num_classes=codebook_indices)
    save_tensor(z, shape_id)
    stats["shapes_dict"][shape_id] = True
    stats["shapes_counter"] += 1


def handle_shape_set_sequentially(shape_set, scores):
    z_set = torch.zeros((8, 8,
                        8, codebook_indices)).to(device)
    for i, shape in enumerate(shape_set):
        z_set += shape * scores[i]
    z_set = z_set / torch.sum(scores)
    print(colored(f'[*] Shape set done', 'yellow'))
    return z_set


def handle_shape_set(scores, shape_id, shape_sets, shape_sets_indices):
    """ Converts each shape set to be a g^3 x n_embed probability distribution over the codebook indices

    Args:
        scores [tensor int[]]: A list of similarity scores for each shape in each shapeset
        shape_id string: Reference shape id
        shape_sets tensor[1,resolution,resolution,resolution][][]: 2D list of sdf tensors for each shape in each shapeset 
        shape_sets_indices int[]: Row index of each shape set
    """
    if (len(shape_sets) == 0):
        return
    for i, shape_set in enumerate(shape_sets):
        shapes_length = shape_set.size()[0]
        print(colored(f'[*]{shapes_length} shape set size', 'blue'))
        if (shapes_length > shapeset_length_threshold):
            z_set = handle_shape_set_sequentially(shape_set, scores=scores[i])
        else:
            z_set = torch.tensor(shape_set).to(device)
            shape_set_scores = scores[i].reshape(
                shape_set.shape[0], 1, 1, 1, 1).to(device)
            total_score = torch.sum(shape_set_scores).to(device)
            # apply weighted average
            z_set = (z_set * shape_set_scores) / total_score
            z_set = torch.sum(z_set, axis=0)
        # Making sure it is a valid probability distribution
        assert torch.all(torch.isclose(
            torch.tensor(1.), torch.sum(z_set, axis=-1)))
        save_tensor(z_set, shape_id, is_shape_set=True,
                    shape_set_row_index=shape_sets_indices[i])
        stats["shape_set_counter"] += 1
        stats["row_indices"][shape_sets_indices[i]] = True

# TODO: Batch process this


def extract_z_shape():
    for i in tqdm(range(len(dataset))):
        element = dataset[i]
        shape_sdf = element["shape_sdf"]
        shape_id = element["shape_id"]

        handle_shape(shape_id, shape_sdf)
        print(colored(f'[*]{shape_id} done', 'blue'))
        save_stats_dict()


def extract_z_shapesets():
    for i in tqdm(range(len(dataset))):
        element = dataset[i]
        shape_sets = element["shape_sets_z"]
        scores = element["shape_sets_scores"]
        shape_id = element["shape_id"]
        shape_sets_indices = element["shape_sets_indices"]
        handle_shape_set(scores, shape_id, shape_sets, shape_sets_indices)
        print(colored(f'[*]{shape_id} shapeset done', 'blue'))
        save_stats_dict()


info = open(f"{dataset_path}/completed_set_8.json",)

dictionary = json.load(info)


print(colored(f'[*]{len(dictionary.keys())} shapes found', 'green'))


def extract_z_shape_set_parallel(i):
    shape_id = dataset.get_id_from_index(i)
    is_done = dictionary.get(shape_id, False)
    if (is_done):
        print(colored(f'[*]{shape_id} already done', 'green'))
        return
    element = dataset[i]
    shape_id = element["shape_id"]
    shape_sets = element["shape_sets_z"]
    scores = element["shape_sets_scores"]
    shape_sets_indices = element["shape_sets_indices"]
    handle_shape_set(scores, shape_id, shape_sets, shape_sets_indices)
    print(colored(f'[*]{shape_id} shapeset done', 'blue'))
    stats["shapes_dict"][shape_id] = True
    stats["shapes_dict"][str(i)] = True
    save_stats_dict()


def parallel_z_set():
    indices = np.arange(len(dataset), dtype=int)
    tokenizer = xmp.Vectorizer(extract_z_shape_set_parallel, num_workers=16)
    tokenizer.process(indices)


completed_shape_set = {}


def find_completed_shape_sets():
    main_directory = f"{root}/{DATA_SET_PATH}/{cat}"
    directories = os.listdir(main_directory)
    print(len(directories))
    for directory in directories:
        files = os.listdir(f"{main_directory}/{directory}")
        if (len(files) >= 3):
            completed_shape_set[directory] = True
    save_dict("completed_set_8.json", completed_shape_set)


# find_completed_shape_sets()
# parallel_z_set()
