import random
import numpy as np
from scripts.validation.validate_3d_numpy_array import validate_3d_numpy_array


AIR_ID = 283


def translate_y(data_3d, y, loop=True):
    """translates the data along the y-axis (up/down)"""
    validate_3d_numpy_array(data_3d)
    if y == 0:
        return np.copy(data_3d)
    if loop:
        return np.roll(data_3d, shift=y, axis=0)

    result = np.zeros_like(data_3d)
    if y > 0:
        result[y:] = data_3d[:-y]
    elif y < 0:
        result[:y] = data_3d[-y:]
    return result


def translate_z(data_3d, z, loop=True):
    """translates the data along the z-axis (north/south)"""
    validate_3d_numpy_array(data_3d)
    if z == 0:
        return np.copy(data_3d)
    if loop:
        return np.roll(data_3d, shift=z, axis=1)

    result = np.zeros_like(data_3d)
    if z > 0:
        result[:, z:, :] = data_3d[:, :-z, :]
    elif z < 0:
        result[:, :z, :] = data_3d[:, -z:, :]
    return result


def translate_x(data_3d, x, loop=True):
    """translates the data along the X-axis (east/west)"""
    validate_3d_numpy_array(data_3d)
    if x == 0:
        return np.copy(data_3d)
    if loop:
        return np.roll(data_3d, shift=x, axis=2)

    result = np.zeros_like(data_3d)
    if x > 0:
        result[:, :, x:] = data_3d[:, :, :-x]
    elif x < 0:
        result[:, :, :x] = data_3d[:, :, -x:]
    return result


def translate(data_3d, x, y, z, loop=True):
    """translates the data"""
    return translate_x(translate_z(translate_y(data_3d, y, loop), z, loop), x, loop)


def random_safe_translate_x(data_3d):
    """applies a random translation along the x-axis (east/west) where only fully empty layers can exceed the
    bounding box."""

    def is_empty_layer(x):
        return np.all(data_3d[:, :, x] == AIR_ID)

    def count_leading_empty_layers():
        empty = 0
        for x in range(width):
            if not is_empty_layer(x):
                break
            empty += 1
        return empty

    def count_trailing_empty_layers():
        empty = 0
        for x in range(width - 1, -1, -1):
            if not is_empty_layer(x):
                break
            empty += 1
        return empty

    width = data_3d.shape[2]
    leading_empty = count_leading_empty_layers()
    trailing_empty = count_trailing_empty_layers()

    valid_shifts = [x for x in range(-leading_empty, trailing_empty + 1)]
    shift = random.choice(valid_shifts)
    return np.roll(data_3d, shift=shift, axis=2)


def random_safe_translate_z(data_3d):
    """applies a random translation along the z-axis (north/south) where only fully empty layers can exceed the
    bounding box."""

    def is_empty_layer(z):
        return np.all(data_3d[:, z, :] == AIR_ID)

    def count_leading_empty_layers():
        empty = 0
        for z in range(length):
            if not is_empty_layer(z):
                break
            empty += 1
        return empty

    def count_trailing_empty_layers():
        empty = 0
        for z in range(length - 1, -1, -1):
            if not is_empty_layer(z):
                break
            empty += 1
        return empty

    length = data_3d.shape[1]
    leading_empty = count_leading_empty_layers()
    trailing_empty = count_trailing_empty_layers()

    valid_shifts = [z for z in range(-leading_empty, trailing_empty + 1)]
    shift = random.choice(valid_shifts)
    return np.roll(data_3d, shift=shift, axis=1)
