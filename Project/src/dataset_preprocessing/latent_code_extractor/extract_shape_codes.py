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
    file_path = f"{dataset_path}/{cat}/{shape_id}/z_shape.pt"
    # file_path = f"{dataset_path}/z_shape.pt"
    if is_shape_set:
        file_path = f"{dataset_path}/{cat}/{shape_id}/z_set_{shape_set_row_index}.pt"
        # file_path = f"{dataset_path}/z_set_{shape_set_row_index}.pt"
    torch.save(input, file_path)


def save_stats_dict():
    with open(f"{dataset_path}/stats.json", 'w') as f:
        # Write the dictionary to the file in JSON format
        json.dump(stats, f)


def handle_shape(shape_id, shape_sdf):
    is_shape_id_done = stats["shapes_dict"].get(shape_id, False)
    if (is_shape_id_done):
        return
    # 8 * 8 * 8
    indices = model.encode_indices(shape_sdf.unsqueeze(0)).squeeze(0)
    # One hot encode
    z = functional.one_hot(indices, num_classes=codebook_indices)
    save_tensor(z, shape_id)
    stats["shapes_dict"][shape_id] = True
    stats["shapes_counter"] += 1


def handle_shape_set(scores, shape_id, shape_set_sdfs, shape_set_indices):
    if (len(shape_set_sdfs) == 0):
        return
    for i, shape_set in enumerate(shape_set_sdfs):
        indices = model.encode_indices(shape_set)
        z_set = functional.one_hot(indices, num_classes=codebook_indices)
        # broadcasting for valid multiplication
        shape_set_scores = scores[i].reshape(
            shape_set.shape[0], 1, 1, 1, 1).to(device)
        total_score = torch.sum(shape_set_scores).to(device)
        # apply weighter average
        z_set = (z_set * shape_set_scores) / total_score
        z_set = torch.sum(z_set, axis=0)
        # Making sure it is a valid probability distribution
        assert torch.all(torch.isclose(
            torch.tensor(1.), torch.sum(z_set, axis=-1)))
        save_tensor(z_set, shape_id, is_shape_set=True,
                    shape_set_row_index=shape_set_indices[i])
        stats["shape_set_counter"] += 1


for i in tqdm(range(len(dataset))):
    element = dataset[i]
    shape_sdf = element["shape_sdf"]
    shape_set_sdfs = element["shape_set_sdfs"]
    shape_set_ids = element["shape_set"]
    scores = element["shape_set_scores"]
    shape_id = element["shape_id"]
    shape_set_indices = element["shape_set_indices"]
    handle_shape(shape_id, shape_sdf)
    # handle_shape_set(scores, shape_id, shape_set_sdfs, shape_set_indices)
save_stats_dict()
