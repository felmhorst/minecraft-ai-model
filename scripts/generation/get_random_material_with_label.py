import numpy as np
from numpy import random


def get_random_material_with_label() -> tuple[int, str]:
    """returns a random block with a descriptive label"""
    material_id = np.random.choice([1, 4])
    # material_label = np.random.choice(["stone" if material_id == 1 else "wooden", ""])
    material_label = np.random.choice(["stone", ""])
    # material_id = 1
    # material_label = ""
    return material_id, material_label
