import sys  # noqadefined
import os  # noqa
sys.path.append(os.path.abspath(os.path.join('..', '')))  # noqa

import shutil
from constants import NRRD_EXTENSION, OBJ_EXTENTION, DEMO_ARGUMENT, DATA_SET_PATH, TEMP_PATH, ERRORS_PATH
from process_one_mesh import process_obj,process_one_obj
#import pdb
import tqdm
#pdb.set_trace()

# Configuration
sdfcommand = './isosurface/computeDistanceField'
mcube_cmd = './isosurface/computeMarchingCubes'
lib_cmd = './isosurface/LIB_PATH'

num_sample = 64 ** 3
bandwidth = 0.1 # snet
sdf_res = 256
expand_rate = 1.3
iso_val = 0.003 # snet
max_verts = 16384
g=0.0 # snet



def construct_full_folder_path(folder_name):
     return f"{DATA_SET_PATH}/{folder_name}"


def construct_full_obj_file_path(folder_name):
    """ Constructs the full path to the nrrd file from the folder
        name in the dataset

    Args:
        file_name: Folder name of the  file relative to the dataset path.

    Returns:
        None

"""
    return f"{construct_full_folder_path(folder_name)}/{folder_name}{OBJ_EXTENTION}"


def obj_to_sdf(file_name,folder_name):
    process_obj(file_name,folder_name)
    process_one_obj(sdfcommand, mcube_cmd, "source %s" % lib_cmd,
                num_sample, bandwidth, sdf_res, expand_rate, file_name, iso_val,
                max_verts,folder_name, ish5=True, normalize=True, g=g, reduce=4)
    



def convert_data_set_to_sdf():
    """ Creates sdf objects for all models found in the dataset 
        Args:
           None

        Returns:
            None
    """
    directories = os.listdir(DATA_SET_PATH)
    with open(TEMP_PATH, "w") as file:
        for index in tqdm.tqdm(range(0, len(directories))):
            directory_name = directories[index]
            folder_name = construct_full_folder_path(directory_name)
            file_name = construct_full_obj_file_path(directory_name)
            obj_to_sdf(file_name)
            file.write(directory_name)
            file.write('\n')
            shutil.rmtree(f'{folder_name}/tmp', ignore_errors=True)


if __name__ == "__main__":
    arguments = sys.argv
    is_demo = len(arguments) > 1 and sys.argv[1] == DEMO_ARGUMENT
    if is_demo:
    #    shutil.rmtree("tmp", ignore_errors=True)
        obj_to_sdf("demo.obj","")
    else:
        convert_data_set_to_sdf()

