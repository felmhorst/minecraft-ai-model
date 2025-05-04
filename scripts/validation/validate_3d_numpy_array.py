import numpy as np


def validate_3d_numpy_array(data):
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError("data_3d must be a 3D numpy array.")