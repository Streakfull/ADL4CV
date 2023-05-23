import pdb
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join('..', '')))  # noqa
from dataset_preprocessing.constants import DATA_SET_PATH, TEXT2SHAPE_DATA_SET_PATH
from datasets.latent_code_extractor import LatentCodeExtractor
from options.base_options import BaseOptions
from encoder.pvqae import PVQVAE
from torch.nn import functional
from tqdm import tqdm
import json


device = "cpu"
if (torch.cuda.is_available()):
    device = "cuda"
root = ".."
dataset_path = f"{root}/{DATA_SET_PATH}"
text2Shape_dataset_path = f"{root}/{TEXT2SHAPE_DATA_SET_PATH}"
cat = "chairs"
shapeset_length_threshold = 20

dataset = LatentCodeExtractor(
    text2Shape_dataset_path, dataset_path, cat=cat, resolution=32)


vq_cfg = f"{root}/configs/pvqae_configs.yaml"
# Chckpoint here
chkpoint = f"{root}/encoder/raw_dataset/logs/pvqae-128-z/ckpt/vqvae_epoch-latest.pth"
options = BaseOptions(config_path=vq_cfg,
                      name="pvqae-extraction", is_train=False)
model = PVQVAE()
model.initialize(options)
model.load_ckpt(chkpoint)

stats = {
    "shapes_dict": {},
    "shapes_counter": 0,
    "shape_set_counter": 0
}
codebook_indices = model.n_embed


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
    torch.save(input, file_path)


def save_stats_dict():
    """Saves the stats that has all relevant info about the process in the dataset path
    """
    with open(f"{dataset_path}/stats.json", 'w') as f:
        # Write the dictionary to the file in JSON format
        json.dump(stats, f)


def handle_shape(shape_id, shape_sdf):
    """ Converts a shape sdf to be a g^3 x n_embed probability distribution over the codebook indices

    Args:
        shape_id string: Shape id
        shape_sdf tensor[1,resolution,resolution,resolution]: sdf grid of the shape
    Saves:
        z tensor[ncubes_per_dim,ncubes_per_dim,ncubes_per_dim,n_embed ]: probability distribution over all possible codebook indices
    """
    is_shape_id_done = stats["shapes_dict"].get(shape_id, False)
    if (is_shape_id_done):
        return
    # 8 * 8 * 8
    # Unsequeezed since the model expects a batch dimension
    indices = model.encode_indices(shape_sdf.unsqueeze(0)).squeeze(0)
    # One hot encode
    z = functional.one_hot(indices, num_classes=codebook_indices)
    save_tensor(z, shape_id)
    stats["shapes_dict"][shape_id] = True
    stats["shapes_counter"] += 1


def handle_shape_set_sequentially(shape_set, scores):
    z_set = torch.zeros((model.ncubes_per_dim, model.ncubes_per_dim,
                        model.ncubes_per_dim, codebook_indices))
    for i, shape in enumerate(shape_set):
        indices = model.encode_indices(shape.unsqueeze(0)).squeeze(0)
        z_shape = functional.one_hot(
            indices, num_classes=codebook_indices) * scores[i]
        z_set += z_shape
    z_set = z_set / torch.sum(scores)


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
        if (shapes_length > shapeset_length_threshold):
            z_set = handle_shape_set_sequentially(shape_set, scores=[i])
        else:
            indices = model.encode_indices(shape_set)
            z_set = functional.one_hot(indices, num_classes=codebook_indices)
            # broadcasting for valid multiplication
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


for i in tqdm(range(len(dataset))):
    element = dataset[i]
    shape_sdf = element["shape_sdf"]
    shape_sets = element["shape_sets_sdfs"]
    shape_set_ids = element["shape_set"]
    scores = element["shape_sets_scores"]
    shape_id = element["shape_id"]
    shape_sets_indices = element["shape_sets_indices"]
    handle_shape(shape_id, shape_sdf)
    handle_shape_set(scores, shape_id, shape_sets, shape_sets_indices)
save_stats_dict()
