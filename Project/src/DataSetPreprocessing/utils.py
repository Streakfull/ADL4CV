import numpy as np
import nrrd
import h5py

# Adapted from https://github.com/kchen92/text2shape


def read_nrrd(nrrd_filename):
    """Reads an NRRD file and returns an RGBA tensor.
    Args:
        nrrd_filename: Filename of the NRRD file.
    Returns:
        voxel_tensor: 4-dimensional voxel tensor with 4 channels (RGBA) where the alpha channel
                is the last channel (aka vx[:, :, :, 3]).
    """
    nrrd_tensor, options = nrrd.read(nrrd_filename)
    assert nrrd_tensor.ndim == 4

    # Convert to float [0, 1]
    voxel_tensor = nrrd_tensor.astype(np.float32) / 255.

    # Move channel dimension to last dimension
    voxel_tensor = np.rollaxis(voxel_tensor, 0, 4)

    # Make the model stand up straight by swapping axes (see README for more information)
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 1)
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 2)

    return voxel_tensor


def export_mesh_to_obj(path, vertices, faces, useFacesPadding=False):
    """
    exports mesh as OBJ
    Args:
      path: output path for the OBJ file
      vertices: Nx3 vertices
      faces: Mx3 faces
    Returns:
        None
    """
    print(f"Creating Obj file for {path} ...")
    # write vertices starting with "v "
    # write faces starting with "f "
    with open(path, "w") as file:
        for vertix in vertices:
            string = np.array2string(vertix, separator=' ', formatter={'float_kind': lambda x: "%.3f" % x}) \
                .strip('[]')
            string = f"v {string}"
            file.write(string)
            file.write('\n')
        if useFacesPadding:
            faces += 1
        for face in faces:
            string = np.array2string(face, separator=' ', formatter={'float_kind': lambda x: "%.3f" % x}) \
                .strip('[]')
            string = f"f {string}"
            file.write(string)
            file.write('\n')


# TODO: Implement this
# def save_sdf_as_numpy(hp_filename):
#     """Reads h5 sdf file and stores it as a 1D numpy text array.
#     Args:
#         hp_filename: Filename of the sdf file.
#     Returns:
#         None
#     """
#     hx = h5py.File(hp_filename,"r")
#     sdf = hx['pc_sdf_sample'][:]
#     #sdf = sdf.reshape((64,64,64))
#     np.savetxt("ok.txt",sdf)
