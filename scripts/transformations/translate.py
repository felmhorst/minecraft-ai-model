import numpy as np
from scripts.validation.validate_3d_numpy_array import validate_3d_numpy_array


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
