
import sys  # noqadefined
import os  # noqa
sys.path.append(os.path.abspath(os.path.join('..', '')))  # noqa
from utils import read_nrrd, export_mesh_to_obj
from constants import NRRD_EXTENSION, OBJ_EXTENTION, DEMO_ARGUMENT, DATA_SET_PATH, TEMP_PATH, ERRORS_PATH
import skimage.measure as measure
import numpy as np
import tqdm



def nrrd_to_mesh(file_name):
    """ Converts an nrrd file from the raw dataset to a mesh object and
        exports the mesh object to the dataset folder defined in constants.py

        Args:
            file_name: Filename of the NRRD file relative to the dataset path.

        Returns:
            None
    """
    voxel_tensor = read_nrrd(file_name)
    alpha_channel = voxel_tensor[:, :, :, 3]
    binary_mask = (alpha_channel > 0).astype(np.uint8)
    try:
        vertices, faces, _, _ = measure.marching_cubes(
            binary_mask, allow_degenerate=True)
        new_file_name = file_name.replace(NRRD_EXTENSION, OBJ_EXTENTION)
        export_mesh_to_obj(new_file_name, vertices, faces)
    except:
        with open(ERRORS_PATH, "w") as file:
            print("ERROR?")
            file.write(file_name)
            file.write('\n')


def construct_full_nrrd_file_path(folder_name):
    """ Constructs the full path to the nrrd file from the folder
        name in the dataset

    Args:
        file_name: Folder name of the NRRD file relative to the dataset path.

    Returns:
        None

"""
    return f"{DATA_SET_PATH}/{folder_name}/{folder_name}{NRRD_EXTENSION}"


def convert_data_set_to_mesh():
    """ Creates mesh objects for all models found in the dataset 
        Args:
           None

        Returns:
            None
    """
    directories = os.listdir(DATA_SET_PATH)
    with open(TEMP_PATH, "w") as file:
        for index in tqdm.tqdm(range(0, len(directories))):
            directory_name = directories[index]
            file_name = construct_full_nrrd_file_path(directory_name)
            nrrd_to_mesh(file_name)
            file.write(directory_name)
            file.write('\n')



if __name__ == "__main__":
    arguments = sys.argv
    is_demo = len(arguments) > 1 and sys.argv[1] == DEMO_ARGUMENT
    if is_demo:
        nrrd_to_mesh("demo.nrrd")
    else:
        convert_data_set_to_mesh()
