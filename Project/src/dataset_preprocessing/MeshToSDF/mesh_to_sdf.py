import sys  # noqadefined
import os  # noqa
sys.path.append(os.path.abspath(os.path.join('..', '')))  # noqa

import shutil
from constants import NRRD_EXTENSION, OBJ_EXTENSION, DEMO_ARGUMENT, DATA_SET_PATH, TEMP_PATH, ERRORS_PATH
from process_one_mesh import process_obj,process_one_obj
#import pdb
import tqdm
import pyblaze.multiprocessing as xmp
import numpy as np
import time
from utils import construct_full_folder_path, construct_full_obj_file_path

#pdb.set_trace()

# Configuration
sdfcommand = './isosurface/computeDistanceField'
mcube_cmd = './isosurface/computeMarchingCubes'
lib_cmd = './isosurface/LIB_PATH'

num_sample = 64 ** 3
bandwidth = 0.1 # snet
sdf_res = 128
expand_rate = 1.3
iso_val = 0.003 # snet
max_verts = 16384
g=0.0 # snet
directories = os.listdir(DATA_SET_PATH)

def obj_to_sdf(file_name,folder_name,index):
    #process_obj(file_name,folder_name)
    process_one_obj(sdfcommand, mcube_cmd, "source %s" % lib_cmd,
                num_sample, bandwidth, sdf_res, expand_rate, file_name, iso_val,
                max_verts,folder_name, index, ish5=True, normalize=True, g=g, reduce=4)
    
def obj_to_sdf_parellel(chunk,index):
    process_one_obj(sdfcommand, mcube_cmd, "source %s" % lib_cmd,
                num_sample, bandwidth, sdf_res, expand_rate, chunk, iso_val,
                max_verts,"test", index, ish5=True, normalize=True, g=g, reduce=4)


def parallel():
    """ Creates sdf objects for all models found in the dataset 
        Args:
           None

        Returns:
            None
    """

    indices = np.arange(len(directories))
    for i in indices:
        directories[i] = construct_full_obj_file_path(directories[i], OBJ_EXTENSION)
    chunked_arrays = np.array_split(directories, len(directories) / 8)
    with open(TEMP_PATH, "w") as file:
        for index in tqdm.tqdm(range(560, len(chunked_arrays))):
            chunk = chunked_arrays[index]
            directory_name = directories[index]
            folder_name = construct_full_folder_path(directory_name)
            obj_to_sdf_parellel(chunk, index)
            file.write(np.array2string(chunk))
            file.write('\n')
            shutil.rmtree(f'{folder_name}/tmp', ignore_errors=True)


if __name__ == "__main__":
    arguments = sys.argv
    is_demo = len(arguments) > 1 and sys.argv[1] == DEMO_ARGUMENT
    if is_demo:
        t0 = time.time()
        obj_to_sdf("demo2.obj","test",10)
        t1 = time.time()
        total = t1-t0
        print(total)
    else:
        t0 = time.time()
        parallel()
        t1 = time.time()
        total = t1-t0
        print(total)


