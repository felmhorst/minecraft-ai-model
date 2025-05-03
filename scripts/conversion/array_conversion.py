import numpy as np


def convert_1d_data_to_3d_array(data, w, h, l):
    """converts a flat array into a 3d array"""
    data_flat = np.array(data, dtype=int)
    data_3d = data_flat.reshape(h, l, w)
    return data_3d


def convert_3d_data_to_1d(data_3d):
    """converts a 3d array into a flat array"""
    return data_3d.flatten()


def get_block(data_3d, x, y, z):
    return data_3d[y, z, x]
