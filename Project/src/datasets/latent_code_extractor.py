from torch.utils.data import Dataset
from ast import literal_eval
from datasets.shape_net import ShapenetDataset
import pandas as pd


class LatentCodeExtractor(Dataset):

    def __init__(self, text2Shape_dir, shape_dir, csv_file_name="similar_phrase_2.csv",  cat="chairs", resolution=64):
        """Dataset class for latent code extractor. This is NOT is the data set class to be used for training
           No phrases/texts are needed here

        Args:
            text2Shape_dir (string): directory containing the similar phrases excel file in the form of 
            tes2Shape_dir/chairs/similar_phrase.csv
            shape_dir (string): directory containing all the shapes and their sdf files
            csv_file_name (str, optional): text2shape csv file name. Defaults to "similar_phrase_2.csv".
            cat (str, optional): category of shapes to be used all|chairs|tables . Defaults to "chairs".
            resolution (int, optional): Resolution of stored sdf files 
        """
        self.shape_dir = shape_dir
        self.text2Shape_dir = text2Shape_dir
        self.shape_dir = shape_dir
        # TODO: Add cat here
        self.shapenet_dataset = ShapenetDataset(
            shape_dir, resolution=resolution, cat=cat)
        self.csv_path = f"{text2Shape_dir}/{cat}/{csv_file_name}"
        self.shape_dict = self.construct_shape_dict()

    def construct_shape_dict(self):
        """Prepares the necessary info for the dataset and sets all the model ids present in the text2Shape dataset, after validating that their sdf files are present.

        Returns:
            shape_dict dict{model_id:{ 
                similar_models:string[][]
                similar_scores:string[][]
            }}:  dictionary containing all the model_ids and all their corresponding shapesets according to the text phrase
                 In total: there are 6589 shapes and 41667 shape sets.
        """
        df = pd.read_csv(self.csv_path)
        df = df[0:100]
        # set rows with the no similar shapes to the empty array
        df.loc[df['similar_model_id'] == "0", "similar_model_id"] = "[]"
        df.loc[df['similar_model_score'] == "0", "similar_model_score"] = "[]"
        # evaluate the the array string representation to actual arrays
        df['similar_model_id'] = df['similar_model_id'].apply(literal_eval)
        df["similar_model_score"] = df['similar_model_score'].apply(
            literal_eval)
        shape_dict = {}
        # Model ids are not unique in the df. pd.todict can't be used here
        for row in df.itertuples(index=False):
            shape_dict.setdefault(
                row.model_id, {"similar_models": [], "similar_scores": []})
            # If similar models are found
            if (len(row.similar_model_id) > 0):
                shape_dict[row.model_id]["similar_models"].append(
                    row.similar_model_id)
                shape_dict[row.model_id]["similar_scores"].append(
                    row.similar_model_score)
        # only consider shapes that have sdf in the dataset
        model_ids = shape_dict.keys()
        self.model_ids = list(
            filter(lambda x: x in model_ids, self.shapenet_dataset.shapes))
        return shape_dict

    def __len__(self):
        return len(self.model_ids)

    def get_sdf_from_model_id(self, model_id):
        return self.shapenet_dataset.get_item_by_id(model_id)

    def __getitem__(self, index):
        """Gets an item for the dataset

        Args:
            index int: index position of the data set

        Returns:
            dict of:
            shape_id string:  The shape id
            shape_set string[][]: 2D array containing the shape ids forming each set
            shape_set_scores float[][]: 2D array containing the similarity score to the reference model
                                        for each shape in each shapeset
            shape_set_sdfs  Tensor[1,resolution,resolution,resolution][] : 2D array of tensors, each tensor represents the sdf representation of each shape in each shape set
            shape_sdf Tensor[1,resolution,resolution,resolution]:  Tensor containing the truncated sdf grid of the reference model

        """
        model_id = self.model_ids[index]
        values = self.shape_dict[model_id]
        shape_sdf = self.get_sdf_from_model_id(model_id)
        similar_models = values["similar_models"]
        similar_shapes_sdfs = [[self.get_sdf_from_model_id(
            model_id) for model_id in shapeset] for shapeset in similar_models]

        item_dict = {
            "shape_id": model_id,
            "shape_set": similar_models,
            "shape_set_scores": values["similar_scores"],
            "shape_set_sdfs": similar_shapes_sdfs,
            "shape_sdf": shape_sdf
        }
        return item_dict

    def name(self):
        return 'LatentCodeExtractorDataset'
