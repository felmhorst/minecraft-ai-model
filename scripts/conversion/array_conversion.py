import numpy as np


def convert_1d_data_to_3d(data, w=16, h=16, l=16) -> np.ndarray:
    """converts a flat array into a 3d array"""
    data_flat = np.array(data, dtype=int)
    data_3d = data_flat.reshape(h, l, w)
    return data_3d


def convert_3d_data_to_1d(data):
    """converts a 3d array into a flat array"""
    data_3d = np.array(data, dtype=int)
    return data_3d.flatten()


def get_block(data_3d, x, y, z):
    return data_3d[y, z, x]
