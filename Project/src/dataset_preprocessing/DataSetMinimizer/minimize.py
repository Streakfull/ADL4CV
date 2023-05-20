from utils import construct_full_obj_file_path
from constants import NRRD_EXTENSION, OBJ_EXTENSION,  DATA_SET_PATH, ERRORS_PATH
import sys  # noqadefined
import os  # noqa
sys.path.append(os.path.abspath(os.path.join('..', '')))


directories = os.listdir(DATA_SET_PATH)


def minimize_dataset():
    for directory in directories:
        deleted_extensions = [NRRD_EXTENSION, OBJ_EXTENSION]
        for extension in deleted_extensions:
            file_name = construct_full_obj_file_path(directory, extension)
            try:
                os.remove(file_name)
            except:
                print(file_name, "not found")


minimize_dataset()
