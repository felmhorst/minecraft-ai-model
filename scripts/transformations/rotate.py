import random
import numpy as np
from scripts.transformations.flip import flip_x, flip_z


def rotate_y(data_3d, rotation):
    """rotates the data by 90, 180, or 270 degrees along the y-axis"""
    # todo: handle block orientations
    if rotation == 0:
        return np.copy(data_3d)
    if rotation == 90:
        return np.transpose(data_3d, (1, 0, 2))[:, :, ::-1]
    if rotation == 180:
        return flip_x(flip_z(data_3d))
    if rotation == 270:
        return np.transpose(data_3d, (1, 0, 2))[:, ::-1, :]
    print('Warning: Can only rotate by 90, 180, 270 degrees!')
    return np.copy(data_3d)


def random_rotate(data_3d):
    """applies a random rotation to the data"""
    rotations = [0, 90, 180, 270]
    rotation = random.choice(rotations)
    return rotate_y(data_3d, rotation)
