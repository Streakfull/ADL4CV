import h5py
import sys
import os  # noqa
# sys.path.append(os.path.abspath(os.path.join('..', '')))


def read_sdf(filename, resolution):
    """Reads an h5 file and returns the 3d sdf grid according to resolution

    Args:
        filename (_type_): full file path with .h5 extension
        resolution (_type_): resolution for sdf grid 32/64

    Returns:
        np.array: 3D np array (resolution x resolution x resolution)
    """
    sdf = h5py.File(filename, "r")
    sdf = sdf['pc_sdf_sample'][:]
    sdf_grid = sdf.reshape((resolution, resolution, resolution))
    return sdf_grid
