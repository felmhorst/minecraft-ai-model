import numpy as np


def resize(data_3d, width, height, length):
    pass


def rotate_data(data_3d, rotation):
    """rotates the data by 90, 180, or 270 degrees"""
    # todo: handle block orientations
    if rotation == 90:
        return np.transpose(data_3d, (1, 0, 2))[:, :, ::-1]
    if rotation == 180:
        return flip_data_x(flip_data_z(data_3d))
    if rotation == 270:
        return np.transpose(data_3d, (1, 0, 2))[:, ::-1, :]
    print('Warning: Can only rotate by 90, 180, 270 degrees!')
    return data_3d


def flip_data_x(data_3d):
    """flips the data along the x-axis"""
    # todo: handle block orientations
    return data_3d[:, :, ::-1]


def flip_data_y(data_3d):
    """flips the data along the y-axis"""
    # todo: handle block orientations
    return data_3d[::-1, :, :]


def flip_data_z(data_3d):
    """flips the data along the z-axis"""
    # todo: handle block orientations
    return data_3d[:, ::-1, :]


def augment_data(data_3d):
    augmented_data = [data_3d, flip_data_x(data_3d), flip_data_z(data_3d)]
    for data in augmented_data.copy():
        for rotation in [90, 180, 270]:
            augmented_data.append(rotate_data(data, rotation))
    return augmented_data
