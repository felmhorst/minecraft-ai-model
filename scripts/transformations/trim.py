import numpy as np


AIR_ID = 283


def trim_y(data_3d):
    """trims the y-layers by removing empty leading and trailing layers"""
    start = 0
    end = data_3d.shape[0]
    while start < end and np.all(data_3d[start] == AIR_ID):
        start += 1
    while end > start and np.all(data_3d[end - 1] == AIR_ID):
        end -= 1
    return data_3d[start:end]


def trim_z(data_3d):
    """trims the z-layers by removing empty leading and trailing layers"""
    start = 0
    end = data_3d.shape[1]
    while start < end and np.all(data_3d[:, start, :] == AIR_ID):
        start += 1
    while end > start and np.all(data_3d[:, end - 1, :] == AIR_ID):
        end -= 1
    return data_3d[:, start:end, :]


def trim_x(data_3d):
    """trims the x-layers by removing empty leading and trailing layers"""
    start = 0
    end = data_3d.shape[2]
    while start < end and np.all(data_3d[:, :, start] == AIR_ID):
        start += 1
    while end > start and np.all(data_3d[:, :, end - 1] == AIR_ID):
        end -= 1
    return data_3d[:, :, start:end]


def trim(data_3d):
    """trims the layers by removing empty leading and trailing layers"""
    return trim_x(trim_z(trim_y(data_3d)))
