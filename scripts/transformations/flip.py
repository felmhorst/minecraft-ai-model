import random
import numpy as np


def flip_x(data_3d):
    """flips the data along the x-axis"""
    # todo: handle block orientations
    return data_3d[:, :, ::-1]


def flip_y(data_3d):
    """flips the data along the y-axis"""
    # todo: handle block orientations
    return data_3d[::-1, :, :]


def flip_z(data_3d):
    """flips the data along the z-axis"""
    # todo: handle block orientations
    return data_3d[:, ::-1, :]


def random_flip(data_3d):
    """applies a random flip to the data"""
    flips = [None, 'x', 'z']
    flip = random.choice(flips)
    if flip == 'x':
        return flip_x(data_3d)
    if flip == 'z':
        return flip_z(data_3d)
    return np.copy(data_3d)
