import random
import numpy as np
from scripts.transformations.flip import flip_x, flip_z


def rotate_y(data_3d, rotation):
    """rotates the data by 90, 180, or 270 degrees along the y-axis"""
    if rotation == 0:
        return np.copy(data_3d)

    rotated = np.empty_like(data_3d)

    for y in range(data_3d.shape[0]):
        xz = data_3d[y]
        if rotation == 90:
            rotated[y] = np.rot90(xz, k=1)
        elif rotation == 180:
            rotated[y] = np.rot90(xz, k=2)
        elif rotation == 270:
            rotated[y] = np.rot90(xz, k=3)
        else:
            print('Warning: Can only rotate by 0, 90, 180, or 270 degrees!')
            return np.copy(data_3d)

    return rotated


def random_rotate(data_3d):
    """applies a random rotation to the data"""
    rotations = [0, 90, 180, 270]
    rotation = random.choice(rotations)
    return rotate_y(data_3d, rotation)
